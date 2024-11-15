import argparse
import os
import time
import copy
import time
import random
import torch
import traceback
import datetime
import itertools
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate, make_batchnorm_stats, make_calibration_dataloader
from metric import make_metric, make_logger
from model import make_model, make_prune_model
from module import save, to_device, process_control, resume, makedir_exist_ok, \
    get_model_profile, summarize_info_list, match_prefix, load, update_model_prof, model_forward, remove_non_picklable_items, check_dense_model, \
    check_calib_saving_info, load_calib_saving_info, save_calib_info, get_layer_order
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
    dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
    if 'mixdataset' in cfg['prune_method']:
        dataset_csr = make_dataset('csr', 'test')
        clm_num_samples = len(dataset_csr['test']) * cfg['batch_size']
        dataset_clm = make_dataset('wikitext', 'test', clm_num_samples, dataset_csr['test'])

    model, tokenizer = make_model(cfg['model_name'])
    cfg['tokenizer'] = tokenizer
    # prepare_cude_events(model)
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    if cfg['model_name'] in ['cnn', 'resnet18', 'wresnet28x2']:
        model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
    model = make_prune_model(model)
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    if 'calib' in cfg['prune_method']:
        print('Running Calibration ...', flush=True)
        cfg['calibration_stage'] = True
        calibration_data_loader = make_calibration_dataloader(tokenizer)
        if check_calib_saving_info() == True:
            load_calib_saving_info(model)
        else:
            run_calibration(model, calibration_data_loader['train'])
            save_calib_info(model)
        if 'flapratio' in cfg['prune_method']:
            from model import HiddenRepresentationPruning
            pruning_module = HiddenRepresentationPruning(cfg, 'flapratio')
            pruning_module.flap_ratio(model, test_logger)
        cfg['calibration_stage'] = False
        print('Calibration Done...', flush=True)
    model_prof = FlopsProfiler(model)
    
    inference_duration = test(data_loader['test'], model, model_prof, metric, test_logger)
    pruned_info_list = get_model_profile('pruned', model_prof)
    onlyprobe_info_list = None
    if 'probe' in cfg['prune_method'] and cfg['onlyprobeinfo'] == True:
        cfg['onlyprobe'] = True
        # change mode to sync to measure the probe flops
        cfg['mode'] = 'sync'
        _ = test(data_loader['test'], model, model_prof, metric, test_logger)
        onlyprobe_info_list = get_model_profile('pruned', model_prof, onlyprobe=True)
    dataset_size = cfg['dataset_size']['test']
    dense_info_list, dense_duration = summarize_info_list(pruned_info_list, inference_duration, test_logger, dataset_size, onlyprobe_info_list)
    
    evaluation = metric.evaluate('test', 'full')
    print('evaluation_for_full', evaluation, flush=True)
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
    
    with torch.no_grad():
        model.eval()
        for i, input in enumerate(data_loader):
            # now, the wikitext and c4 datsets used for calibration are clm tasks
            input_size = input['labels'].size(0)
            input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                    'labels': input['labels']}
            input = to_device(input, cfg['device'])
            output = model(**input)
            input_ = {'target': input['labels']}
            output_ = {'target': output['logits'], 'loss': output['loss']}
    return


def identify_pad_tokens(input):
    pad_tokens = input['input_ids'] == cfg['pad_token_id'] 
    no_padding = (~pad_tokens).all()
    # if there is padding, need to zero out the padding token
    if no_padding == False:
        cfg['pad_tokens'] = pad_tokens
        # cfg['non_pad_tokens'] = ~pad_tokens.to(cfg['data_type'])
        # avoid overflow
        cfg['nonpad_tokens_denominator'] = torch.sum(~cfg['pad_tokens'], dim=0).unsqueeze(1) + 1e-3
    else:
        cfg['pad_tokens'] = None
        # cfg['non_pad_tokens'] = None
        cfg['nonpad_tokens_denominator'] = None

    # print('attentionmasdk', input['attention_mask'])
    # flipped = torch.flip(input['attention_mask'], dims=[1])
    # print("Flipped Tensor:", flipped)
    # # Find the index of the first '1' in each flipped row (now the last '1' in the original)
    # last_one_indices = torch.argmax(flipped, dim=1)
    # print('last_one_indices', last_one_indices)
    # # Correct the indices to reflect their positions in the original tensor
    # num_nonpad_tokens = flipped.size(1) - last_one_indices
    # cfg['num_nonpad_tokens'] = num_nonpad_tokens
    # print("cfg['num_nonpad_tokens'] ", cfg['num_nonpad_tokens'] )
    return

def test(data_loader, model, model_prof, metric, logger):
    torch.cuda.empty_cache()
    start_time = time.time()

    with torch.no_grad():
        
        model.train(False)
        start_time = time.time()
        inference_duration = 0

        # warm up pytorch
        data_loader_iter = iter(data_loader)
        input = next(data_loader_iter)
        identify_pad_tokens(input)
        # TODO: delete this
        cfg['input_ids'] = input['input_ids']
        # print('start input_ids', input['input_ids'], input['input_ids'].size())
        cfg['cur_batch_index'] += 1
        if cfg['task_name'] in ['clm']:
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
            input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                    'labels': input['labels']}
            input = to_device(input, cfg['device'])
            output = model(**input)
            input_ = {'input_indices': input_indices, 'target': input['labels'], 'correct_labels': correct_labels}
            output_ = {'target': output['logits'], 'loss': output['loss']}
        elif cfg['task_name'] in ['mix']:
            input_size = input['labels'].size(0)
            input_indices = input['input_indices']
            correct_labels = input['correct_labels']
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

        # return


        model_prof.start_profile()
        model_prof.reset_profile()
        update_model_prof(model_prof)
        torch.cuda.cudart().cudaProfilerStart()
        for i, input in enumerate(data_loader):
            cfg['cur_batch_index'] += 1
            print('cur_batch_index', cfg['cur_batch_index'])
            identify_pad_tokens(input)
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
            elif cfg['task_name'] in ['mix']:
                # first half for csr, second half for clm
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
                print('evaluation_for_batch', evaluation, flush=True)
                logger.append(evaluation, 'test', input_size)

            for name, module in model.named_modules():
                for attr_name in dir(module):
                    # Check if the attribute name contains 'mean_intersection_ratio'
                    if 'attn_sign_match_percentage' in attr_name or 'attn_l2_magnitude_ratio' in attr_name or 'attn_cosine_similarity' in attr_name\
                        or 'mlp_sign_match_percentage' in attr_name or 'mlp_l2_magnitude_ratio' in attr_name or 'mlp_cosine_similarity' in attr_name:
                        # Retrieve the attribute value
                        attr_value = getattr(module, attr_name)
                        # Print the module name and attribute name
                        # print('name', name, 'attr_name', attr_name, 'attr_value', attr_value)
                        # Append the attribute to the logger
                        logger.append({f'{name}_{attr_name}': attr_value}, 'test')
                        print('name', name, 'attr_name', attr_name)
                    if 'diff_ratio' in attr_name:
                        # Retrieve the attribute value
                        attr_value = getattr(module, attr_name)
                        
                            # Append the attribute to the logger
                        logger.append({f'{name}_{attr_name}': attr_value}, 'test')
                        print('name', name, attr_name, attr_value)
                    if 'cur_select_indices' in attr_name:
                        # Retrieve the attribute value
                        attr_value = getattr(module, attr_name)
                        # Append the attribute to the logger
                        logger.accumulate({f'{name}_{attr_name}': attr_value}, 'test')
                        # print('name', name, attr_name, attr_value)
                        
                    # Check if the attribute name contains 'mean_intersection_ratio'
                    

            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                batch_time = (time.time() - start_time) / (i + 1)
                exp_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                print('running_info', info)

        if 'recordspeed' in cfg['prune_method']:
            cur_attn_inference_duration_list = []
            cur_mlp_inference_duration_list = []
            for name, module in model.named_modules():
                for attr_name in dir(module):
                    if 'cur_attn_inference_duration' in attr_name:
                        # Retrieve the attribute value
                        # if 'opt-13b' in cfg['model_name'] and get_layer_order(name) >= 20:
                        #     continue
                        attr_value = getattr(module, attr_name)
                        cur_attn_inference_duration_list.append(attr_value)
                        # logger.append({f'{name}_{attr_name}': attr_value}, 'test')
                        print('name', name, attr_name, attr_value)
                    if 'cur_mlp_inference_duration' in attr_name:
                        # diff gpu cannt measure the inference time correctly
                        # if 'opt-13b' in cfg['model_name'] and get_layer_order(name) >= 20:
                        #     continue
                        # Retrieve the attribute value
                        attr_value = getattr(module, attr_name)
                        cur_mlp_inference_duration_list.append(attr_value)
                        # logger.append({f'{name}_{attr_name}': attr_value}, 'test')
                        print('name', name, attr_name, attr_value)

            mean_cur_attn_inference_duration = sum(cur_attn_inference_duration_list)/len(cur_attn_inference_duration_list)
            print('length', len(cur_attn_inference_duration_list))
            mean_cur_mlp_inference_duration = sum(cur_mlp_inference_duration_list)/len(cur_mlp_inference_duration_list)
            logger.append({f'attn_inference_duration': mean_cur_attn_inference_duration}, 'test')
            logger.append({f'mlp_inference_duration': mean_cur_mlp_inference_duration}, 'test')
            print('mean_cur_attn_inference_duration', mean_cur_attn_inference_duration)
            print('mean_cur_mlp_inference_duration', mean_cur_mlp_inference_duration)
            print('mean_inference_duration', inference_duration/len(data_loader))
            print('inference_duration', inference_duration)

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

