import argparse
import os
import copy
import time
import torch
import datetime
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate, make_batchnorm_stats
from metric import make_metric, make_logger
from model import make_model, make_prune_model
from module import save, to_device, process_control, resume, makedir_exist_ok, \
    record_pruing_info, get_model_profile, summarize_info_list, match_prefix
from deepspeed.profiling.flops_profiler import FlopsProfiler


cudnn.benchmark = True
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
    model_path = os.path.join('output', 'model')
    result_path = os.path.join('output', 'result')
    model_tag_path = match_prefix(model_path)
    checkpoint_path = os.path.join(model_tag_path, 'checkpoint')
    best_path = os.path.join(model_tag_path, 'best')
    
    model, tokenizer = make_model(cfg['model_name'])
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    # result = resume(os.path.join(best_path, 'model'))
    result = resume(os.path.join(checkpoint_path, 'model'))

    if 'epoch' in result:
        cfg['epoch'] = result['epoch']
    else:
        cfg['epoch'] = 0


    # test FL one
    # if cfg['model_name'] in ['cnn', 'resnet18', 'wresnet28x2']:
    #     result['model_state_dict'] = result['server'].server_model_state_dict
    model.load_state_dict(result['model_state_dict'])
    if cfg['model_name'] in ['cnn', 'resnet18', 'wresnet28x2']:
        model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
    model_prof = FlopsProfiler(model)
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test(data_loader['test'], model, model_prof, copy.deepcopy(metric), test_logger)
    dense_info_list, dense_duration = get_model_profile('dense', model_prof)


    model, tokenizer = make_model(cfg['model_name'])
    model.load_state_dict(result['model_state_dict'])    
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    if cfg['model_name'] in ['cnn', 'resnet18', 'wresnet28x2']:
        model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
    model = make_prune_model(model)
    model_prof = FlopsProfiler(model)
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test(data_loader['test'], model, model_prof, copy.deepcopy(metric), test_logger)
    pruned_info_list, pruned_duration = get_model_profile('pruned', model_prof)
    
    # print('dense_info_list', dense_info_list[0], dense_info_list[1])
    batch_num = len(data_loader['test'])
    summarize_info_list(dense_info_list, pruned_info_list, dense_duration, pruned_duration, batch_num, test_logger)
    # result = resume(os.path.join(checkpoint_path, 'model'))

    # thread lock bug
    test_logger.writer = None
    result = {'cfg': cfg, 'epoch': cfg['epoch'], 'logger': {'test': test_logger}}
    # result = {'cfg': cfg, 'epoch': cfg['epoch']}
    save(result, os.path.join(result_path, cfg['model_tag']))
    return


def test(data_loader, model, model_prof, metric, logger):
    start_time = time.time()
    with torch.no_grad():
        model_prof.start_profile()
        
        model.train(False)
        for i, input in enumerate(data_loader):
            if cfg['task_name'] in ['s2s', 'sc', 'clm', 'csr']:
                input_size = input['labels'].size(0)
                input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                         'labels': input['labels']}
                input = to_device(input, cfg['device'])
                output = model(**input)
                input_ = {'target': input['labels']}
                output_ = {'target': output['logits'], 'loss': output['loss']}
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
            # print('evaluation', evaluation)
            logger.append(evaluation, 'test', input_size)
            record_pruing_info(model, logger)
            # print('output', output_)
            # break
            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                batch_time = (time.time() - start_time) / (i + 1)
                exp_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                print('running_info', info)
        evaluation = metric.evaluate('test', 'full')
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
        logger.append(info, 'test')
        print(logger.write('test', metric.metric_name['test']), flush=True)
        model_prof.stop_profile()
    return


if __name__ == "__main__":
    main()





#    def _check_target_module_exists(target_modules, key):
#         import re
#         if isinstance(target_modules, str):
#             target_module_found = re.fullmatch(target_modules, key)
#         else:
#             # target_module_found = any(key.endswith(target_key) for target_key in target_modules)
#             target_module_found = any(key.endswith(target_key) for target_key in target_modules)

#             # for target_key in target_modules:
#             #     if key.endswith(target_key):
#             #         target_module_found = True
#             #         break

#         return target_module_found
    
#     # Define the forward hook function
#     def forward_hook(module, input, output):
#         print(f"Inside the forward hook of {module.key}")
#         # You can add more functionality here as needed
#         print('input', input[0][0][0])
#         print('output', output[0][0][0])
#         print('weight', module.weight[0][0][0], module.weight[1][0][0], module.stride, module.padding, module.dilation, module.groups, flush=True)
#         print('bias', module.bias, flush=True)

#     # Iterate through the model's layers and register the hook
#     for name, module in model.named_modules():
#         print('name', name)
#         if _check_target_module_exists(['.conv1', '.chortcut', '.conv2'], name):
#             module.key = name
#             module.register_forward_hook(forward_hook)