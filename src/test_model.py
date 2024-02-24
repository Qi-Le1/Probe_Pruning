import argparse
import os
import time
import copy
import time
import random
import torch
import traceback
import datetime
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate, make_batchnorm_stats, make_calibration_dataloader
from metric import make_metric, make_logger
from model import make_model, make_prune_model
from module import save, to_device, process_control, resume, makedir_exist_ok, \
    record_pruing_info, get_model_profile, summarize_info_list, match_prefix, load
from deepspeed.profiling.flops_profiler import FlopsProfiler



cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    # Get the name of the current file
    current_file_name = os.path.basename(__file__)
    print(f"The current file name is {current_file_name}")
    # You can also use conditions to differentiate behavior
    cfg['python_file'] = current_file_name
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    result_path = os.path.join('output', 'result')
    makedir_exist_ok(result_path)

    cfg['epoch'] = 0 
    dense_name_list = cfg['model_tag'].split('_')
    # batch size
    dense_name_list[4] = '10'
    # metric
    dense_name_list[6] = '0'
    # prune_name
    dense_name_list[7] = 'dense'
    # cust_tgt_modules
    dense_name_list[8] = 'None'
    dense_model_path = os.path.join(result_path, '_'.join(dense_name_list))
    if not os.path.exists(dense_model_path):
        dense_model_path = os.path.join(result_path, 'dense', '_'.join(dense_name_list))
    dense_res = load(dense_model_path)
    dense_info_list, dense_duration = dense_res['dense_info_list'], dense_res['dense_duration']

    dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
    model, tokenizer = make_model(cfg['model_name'])
    if 'calib' in cfg['prune_method']:
        calibration_data_loader = make_calibration_dataloader(tokenizer)
        cfg['calibration_stage'] = True
        run_calibration(model, calibration_data_loader['train'])
        cfg['calibration_stage'] = False


    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    if cfg['model_name'] in ['cnn', 'resnet18', 'wresnet28x2']:
        model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
    model = make_prune_model(model)
    model_prof = FlopsProfiler(model)
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test(data_loader['test'], model, model_prof, metric, test_logger)
    pruned_info_list, pruned_duration = get_model_profile('pruned', model_prof)
    
    # print('dense_info_list', dense_info_list[0], dense_info_list[1])
    summarize_info_list(dense_info_list, pruned_info_list, dense_duration, pruned_duration, test_logger)
    # thread lock bug
    test_logger.writer = None
    result = {'cfg': cfg, 'epoch': cfg['epoch'], 'logger': {'test': test_logger},\
              'dense_info_list': dense_info_list, 'pruned_info_list': pruned_info_list, \
              'dense_duration': dense_duration, 'pruned_duration': pruned_duration, 'dataset_size': cfg['dataset_size']['test']}
    save(result, os.path.join(result_path, cfg['model_tag']))
    return


def run_calibration(model, data_loader):
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            if cfg['task_name'] in ['s2s', 'sc', 'clm']:
                input_size = input['labels'].size(0)
                input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                        'labels': input['labels']}
                input = to_device(input, cfg['device'])
                output = model(**input)
                input_ = {'target': input['labels']}
                output_ = {'target': output['logits'], 'loss': output['loss']}
            elif cfg['task_name'] in ['csr']:
                input_size = input['labels'].size(0)
                input_indices = input['input_indices']
                correct_labels = input['correct_labels']
                # print('input', input)
                input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                        'labels': input['labels']}
                input = to_device(input, cfg['device'])
                output = model(**input)
                input_ = {'input_indices': input_indices, 'target': input['labels'], 'correct_labels': correct_labels}
                output_ = {'target': output['logits'], 'loss': output['loss']}
                # print('outputloss', output['loss'])
            else:
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(**input)
                input_ = {'target': input['target']}
                output_ = {'target': output['target'], 'loss': output['loss']}
            if cfg['task_name'] == 's2s':
                output_['generate'] = model.generate(input_ids=input["input_ids"],
                                                    max_new_tokens=cfg['max_new_tokens'])
            elif cfg['task_name'] == 'clm':
                if cfg['data_name'] in ['dolly']:
                    output_['generate'] = model.generate(input_ids=input["input_ids"],
                                                        attention_mask=input["attention_mask"],
                                                        max_new_tokens=cfg['max_new_tokens'],
                                                        eos_token_id=cfg['pad_token_id'],
                                                        no_repeat_ngram_size=2)
    return



def test(data_loader, model, model_prof, metric, logger):
    # print("Debug 12.01: Test logger created", flush=True)
    start_time = time.time()
    with torch.no_grad():
        model_prof.start_profile()
        model.train(False)
        # print("Debug 12.011: Test logger created", flush=True)
        for i, input in enumerate(data_loader):
            # print("Debug 12.1: Test logger created", flush=True)
            if cfg['task_name'] in ['s2s', 'sc', 'clm']:
                input_size = input['labels'].size(0)
                input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                        'labels': input['labels']}
                input = to_device(input, cfg['device'])
                output = model(**input)
                input_ = {'target': input['labels']}
                output_ = {'target': output['logits'], 'loss': output['loss']}
            elif cfg['task_name'] in ['csr']:
                input_size = input['labels'].size(0)
                input_indices = input['input_indices']
                correct_labels = input['correct_labels']
                # print('input', input)
                input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                        'labels': input['labels']}
                input = to_device(input, cfg['device'])
                output = model(**input)
                input_ = {'input_indices': input_indices, 'target': input['labels'], 'correct_labels': correct_labels}
                output_ = {'target': output['logits'], 'loss': output['loss']}
                # print('outputloss', output['loss'])
            else:
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(**input)
                input_ = {'target': input['target']}
                output_ = {'target': output['target'], 'loss': output['loss']}
            if cfg['task_name'] == 's2s':
                output_['generate'] = model.generate(input_ids=input["input_ids"],
                                                    max_new_tokens=cfg['max_new_tokens'])
            elif cfg['task_name'] == 'clm':
                if cfg['data_name'] in ['dolly']:
                    output_['generate'] = model.generate(input_ids=input["input_ids"],
                                                        attention_mask=input["attention_mask"],
                                                        max_new_tokens=cfg['max_new_tokens'],
                                                        eos_token_id=cfg['pad_token_id'],
                                                        no_repeat_ngram_size=2)
            metric.add('test', input_, output_)
            evaluation = metric.evaluate('test', 'batch', input_, output_)
            print('evaluation_for_batch', evaluation)
            logger.append(evaluation, 'test', input_size)
            record_pruing_info(model, logger)
            # if i == 50:
            # if i == 10:
            #     break
            # break
            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                batch_time = (time.time() - start_time) / (i + 1)
                exp_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                print('running_info', info)
        evaluation = metric.evaluate('test', 'full')
        print('evaluation_for_full', evaluation)
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
        logger.append(info, 'test')
        print(logger.write('test', metric.metric_name['test']), flush=True)
        model_prof.stop_profile()
        print("Debug 12.2: Test logger created", flush=True)
    return


if __name__ == "__main__":
    main()

