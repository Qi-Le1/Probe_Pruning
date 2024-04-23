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

iterate_small_samples = False
# iterate_small_samples = True

cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
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

# def prepare_cude_events(model):
#     if 'llama-2' in cfg['model_name']:
#         for i in range(model.config.num_hidden_layers):
#             cfg[f'cuda_events_mlp_{i}'] = torch.cuda.Event()
        

def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    result_path = os.path.join('output', 'result')
    makedir_exist_ok(result_path)
    if check_dense_model() is None:
        print('No dense model found, will not print out the dense model info')
    cfg['epoch'] = 0 
    dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
    model, tokenizer = make_model(cfg['model_name'])
    # prepare_cude_events(model)
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    if cfg['model_name'] in ['cnn', 'resnet18', 'wresnet28x2']:
        model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
    model = make_prune_model(model)
    if 'calib' in cfg['prune_method']:
        print('Running Calibration ...')
        calibration_data_loader = make_calibration_dataloader(tokenizer)
        cfg['calibration_stage'] = True
        run_calibration(model, calibration_data_loader['train'])
        cfg['calibration_stage'] = False
        print('Calibration Done...')
    model_prof = FlopsProfiler(model)
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    inference_duration = test(data_loader['test'], model, model_prof, metric, test_logger)
    pruned_info_list = get_model_profile('pruned', model_prof)
    onlyprobe_info_list = None
    if 'probe' in cfg['prune_method'] and cfg['onlyprobeinfo'] == True:
        cfg['onlyprobe'] = True
        # change mode to sync to measure the probe flops
        cfg['mode'] = 'sync'
        inference_duration = test(data_loader['test'], model, model_prof, metric, test_logger)
        onlyprobe_info_list = get_model_profile('pruned', model_prof, onlyprobe=True)
    dense_info_list, dense_duration = summarize_info_list(pruned_info_list, inference_duration, test_logger, onlyprobe_info_list)
    
    evaluation = metric.evaluate('test', 'full')
    print('evaluation_for_full', evaluation)
    # thread lock bug
    test_logger.save(False)
    test_logger.writer = None
    remove_non_picklable_items(cfg)
    result = {'cfg': cfg, 'epoch': cfg['epoch'], 'logger': {'test': test_logger},\
              'dense_info_list': dense_info_list, 'pruned_info_list': pruned_info_list, \
              'dense_duration': dense_duration, 'pruned_duration': inference_duration, 'dataset_size': cfg['dataset_size']['test']}
    save(result, os.path.join(result_path, cfg['model_tag']))
    return


def run_calibration(model, data_loader):
    from model import HiddenRepresentationPruning
    with torch.no_grad():
        model.eval()
        for i, input in enumerate(data_loader):
            print('calibration', i, input['input_ids'].shape, flush=True)
            # if cfg['task_name'] in ['s2s', 'sc', 'clm']:
            # now, the wikitext and c4 datsets used for calibration are clm tasks
            input_size = input['labels'].size(0)
            input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                    'labels': input['labels']}
            input = to_device(input, cfg['device'])
            output = model(**input)
            input_ = {'target': input['labels']}
            output_ = {'target': output['logits'], 'loss': output['loss']}
            if iterate_small_samples:
                if i == 100:
                    break
        if 'flapratio' in cfg['prune_method']:
            pruning_module = HiddenRepresentationPruning(cfg, 'flapratio')
            pruning_module.flap_ratio(model)
        elif 'gridratio' in cfg['prune_method']:
            pruning_module = HiddenRepresentationPruning(cfg, 'gridratio')
            pruning_module.grid_ratio(model)
        elif 'gridsearch' in cfg['prune_method']:
            pruning_module = HiddenRepresentationPruning(cfg, 'gridsearch')
            pruning_module.grid_search(model)
            # global_determine_ratio(model, if_log=False)
            # global_determine_ratio(model, if_log=True)
            # raise ValueError('Global Determine Ratio Done...')
        # raise ValueError('Calibration Done...')
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
            # record_pruing_info(model, logger)
            # break
            if iterate_small_samples:
                if i == 100:
                    break
            for name, module in model.named_modules():
                for attr_name in dir(module):
                    # Check if the attribute name contains 'mean_intersection_ratio'
                    if 'mean_intersection_ratio' in attr_name:
                        # Retrieve the attribute value
                        attr_value = getattr(module, attr_name)
                        # Print the module name and attribute name
                        # print('name', name, 'attr_name', attr_name, 'attr_value', attr_value)
                        # Append the attribute to the logger
                        logger.append({f'{name}_{attr_name}': attr_value}, 'test')
            
            # if i == 50:
            # if i == 100:
            #     break
            # if i == 10:
            #     break
            # break
            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                batch_time = (time.time() - start_time) / (i + 1)
                exp_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                print('running_info', info)
            # if i == 3:
            #     break
        # torch.cuda.synchronize()
        # inference_duration = time.time() - start_time
        # print('inference_duration', inference_duration)
        if cfg['onlyprobe'] == False: 
            evaluation = metric.evaluate('test', 'full')
            print('evaluation_for_full', evaluation)
            logger.append(evaluation, 'test')
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
            logger.append(info, 'test')
            print(logger.write('test', metric.metric_name['test']), flush=True)
        model_prof.stop_profile()

        inference_time = model.model.model.inference_time
        print('inference_time', inference_time)
        for name, module in model.named_modules():
            for attr_name in dir(module):
                # Check if the attribute name contains 'mean_intersection_ratio'
                if 'position_distribution' in attr_name:
                    # Retrieve the attribute value
                    attr_value = getattr(module, attr_name)
                    if len(attr_value) > 0:
                        # Print the module name and attribute name
                        # print('name', name, 'attr_name', attr_name, 'attr_value', attr_value)
                        # Append the attribute to the logger
                        logger.append({f'{name}_{attr_name}': attr_value}, 'test')

        print("Debug 12.2: Test logger created", flush=True)
        torch.cuda.cudart().cudaProfilerStop()
    return inference_duration


if __name__ == "__main__":
    main()

