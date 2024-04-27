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
    record_pruing_info, get_model_profile, summarize_info_list, match_prefix, load, update_model_prof, model_forward, remove_non_picklable_items, check_dense_model
from deepspeed.profiling.flops_profiler import FlopsProfiler
import matplotlib.pyplot as plt

cudnn.benchmark = False

parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
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
    if check_dense_model() is None:
        print('No dense model found, will not print out the dense model info')
    cfg['epoch'] = 0 
    model, tokenizer = make_model(cfg['model_name'])
    # prepare_cude_events(model)
    # data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    model = make_prune_model(model)
    if 'calib' not in cfg['prune_method']:
        raise ValueError('Calibration is not done, please check the prune method')
    print('Running Calibration ...')
    calibration_data_loader = make_calibration_dataloader(tokenizer)
    cfg['calibration_stage'] = True
    run_calibration(model, calibration_data_loader['train'])
    cfg['calibration_stage'] = False
    print('Calibration Done...')
    model_prof = FlopsProfiler(model)
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    inference_duration = test(calibration_data_loader['train'], model, model_prof, metric, test_logger)
    pruned_info_list = get_model_profile('pruned', model_prof)
    dataset_size = cfg['dataset_size']['test']
    dense_info_list, dense_duration = summarize_info_list(pruned_info_list, inference_duration, test_logger, dataset_size)
    
    evaluation = metric.evaluate('test', 'full')
    print('evaluation_for_full', evaluation)
    # thread lock bug
    test_logger.save(False)
    test_logger.writer = None
    remove_non_picklable_items(cfg)
    result = {'cfg': cfg, 'epoch': cfg['epoch'], 'logger': {'test': test_logger},\
              'dense_info_list': dense_info_list, 'pruned_info_list': pruned_info_list, \
              'dense_duration': dense_duration, 'pruned_duration': inference_duration, 'dataset_size': dataset_size}
    save(result, os.path.join(result_path, cfg['model_tag']))
    return


def run_calibration(model, data_loader):
    from model import HiddenRepresentationPruning
    with torch.no_grad():
        model.eval()
        for i, input in enumerate(data_loader):
            print('calibration_batch_index', i)
            # now, the wikitext and c4 datsets used for calibration are clm tasks
            input_size = input['labels'].size(0)
            input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                    'labels': input['labels']}
            input = to_device(input, cfg['device'])
            output = model(**input)
            input_ = {'target': input['labels']}
            output_ = {'target': output['logits'], 'loss': output['loss']}
        # if 'gridsearch' in cfg['prune_method']:
        #     pruning_module = HiddenRepresentationPruning(cfg, 'gridsearch')
        #     pruning_module.grid_search(model)
        # else:
        #     raise ValueError('gridsearch needs to be in the prune method')
    return



def test(data_loader, model, model_prof, metric, logger):
    start_time = time.time()
    with torch.no_grad():
        model.train(False)
        start_time = time.time()
        inference_duration = 0

        # warm up pytorch
        data_loader_iter = iter(data_loader)
        input = next(data_loader_iter)
        cfg['cur_batch_index'] += 1
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
        else:
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(**input)
            input_ = {'target': input['target']}
            output_ = {'target': output['target'], 'loss': output['loss']}
        torch.cuda.synchronize()

        # start_time = time.time()
        model_prof.start_profile()
        model_prof.reset_profile()
        update_model_prof(model_prof)
        torch.cuda.cudart().cudaProfilerStart()
        for i, input in enumerate(data_loader):
            cfg['cur_batch_index'] += 1
            
            # if cfg['logger_detailed_info']:
            print('cur_batch_index', cfg['cur_batch_index'])
            if cfg['task_name'] in ['s2s', 'sc', 'clm']:
                input_size = input['labels'].size(0)
                input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                        'labels': input['labels']}
                input = to_device(input, cfg['device'])
                output, inference_duration = model_forward(model, input, inference_duration, i)
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
                output, inference_duration = model_forward(model, input, inference_duration, i)
                input_ = {'input_indices': input_indices, 'target': input['labels'], 'correct_labels': correct_labels}
                output_ = {'target': output['logits'], 'loss': output['loss']}
            else:
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output, inference_duration = model_forward(model, input, inference_duration, i)
                input_ = {'target': input['target']}
                output_ = {'target': output['target'], 'loss': output['loss']}

            if cfg['onlyprobe'] == False: 
                metric.add('test', input_, output_)
                evaluation = metric.evaluate('test', 'batch', input_, output_)
                print('evaluation_for_batch', evaluation)
                logger.append(evaluation, 'test', input_size)

            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                batch_time = (time.time() - start_time) / (i + 1)
                exp_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                print('running_info', info)

        if cfg['onlyprobe'] == False: 
            evaluation = metric.evaluate('test', 'full')
            print('evaluation_for_full', evaluation)
            logger.append(evaluation, 'test')
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
            logger.append(info, 'test')
            print(logger.write('test', metric.metric_name['test']), flush=True)
        model_prof.stop_profile()
        torch.cuda.cudart().cudaProfilerStop()
    return inference_duration


if __name__ == "__main__":
    main()

