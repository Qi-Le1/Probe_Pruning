import argparse
import os
import time
import copy
import time
import random
import torch
import fnmatch
import traceback
import datetime
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate, make_batchnorm_stats
from metric import make_metric, make_logger
from model import make_model, make_prune_model
from module import save, to_device, process_control, resume, makedir_exist_ok, \
    record_pruing_info, get_model_profile, summarize_info_list, match_prefix, update_model_prof, model_forward
from deepspeed.profiling.flops_profiler import FlopsProfiler
from lm_eval import tasks
from lm_eval import evaluator



cudnn.benchmark = False
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
    model, tokenizer = make_model(cfg['model_name'])
    # if cfg['task_name'] == 'csr':
    #     model_prof = FlopsProfiler(model)
    #     test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    #     def pattern_match(patterns, source_list):
    #         task_names = set()
    #         for pattern in patterns:
    #             for matching in fnmatch.filter(source_list, pattern):
    #                 task_names.add(matching)
    #         return list(task_names)
    #     # "boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"
    #     task_names = pattern_match(cfg['data_name'], tasks.ALL_TASKS)
    #     model_args = cfg['model_name']
    #     limit = None 
    #     if "70b" in cfg['model_name'] or "65b" in cfg['model_name']:
    #         limit = 2000
    #     accelerate=False
    #     if "30b" in args.model or "65b" in args.model or "70b" in args.model:
    #         accelerate=True
    #     if accelerate:
    #         model_args = f"cfg['model_name'],use_accelerate=True"
    #     model_prof.start_profile()
    #     results = evaluator.simple_evaluate(
    #         model="hf-causal-experimental",
    #         model_args=model_args,
    #         tasks=task_names,
    #         num_fewshot=0,
    #         batch_size=None,
    #         device=None,
    #         no_cache=True,
    #         limit=limit,
    #         description_dict={},
    #         decontamination_ngrams_path=None,
    #         check_integrity=False,
    #         pretrained_model=model,
    #         tokenizer=tokenizer, 
    #         add_special_tokens=False
    #     )
    #     model_prof.stop_profile()
    #     accuracy = results[cfg['data_name']]['acc'] * 100
    #     test_logger.append({'Accuracy': accuracy}, 'test', 1)
    #     dense_info_list, dense_duration = get_model_profile('dense', model_prof)
    # else:
    dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    if cfg['model_name'] in ['cnn', 'resnet18', 'wresnet28x2']:
        model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
    model = make_prune_model(model)
    model_prof = FlopsProfiler(model)
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    inference_duration = test(data_loader['test'], model, model_prof, metric, test_logger)
    dense_info_list = get_model_profile('dense', model_prof)
    print('dense_info_list', dense_info_list)
    print('inference_duration', inference_duration)
    # thread lock bug
    test_logger.writer = None
    result = {'cfg': cfg, 'epoch': cfg['epoch'], 'logger': {'test': test_logger},\
              'dense_info_list': dense_info_list, 'dense_duration': inference_duration}

    save(result, os.path.join(result_path, cfg['model_tag']))
    return




def test(data_loader, model, model_prof, metric, logger):   
    with torch.no_grad():
        model_prof.start_profile()
        update_model_prof(model_prof)
        model.train(False)
        start_time = time.time()
        inference_duration = 0
        for i, input in enumerate(data_loader):
            print("Debug 12.1: Test logger created", flush=True)
            if cfg['task_name'] in ['s2s', 'sc', 'clm']:
                input_size = input['labels'].size(0)
                input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                        'labels': input['labels']}
                input = to_device(input, cfg['device'])
                output, inference_duration = model_forward(model, input, inference_duration)
                input_ = {'target': input['labels']}
                output_ = {'target': output['logits'], 'loss': output['loss']}
            elif cfg['task_name'] in ['csr']:
                input_size = input['labels'].size(0)
                input_indices = input['input_indices']
                correct_labels = input['correct_labels']
                # print('input', input)
                # if 'text' in input:
                # print('input_text', input['text_seq'])
                input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                        'labels': input['labels']}
                input = to_device(input, cfg['device'])
                output, inference_duration = model_forward(model, input, inference_duration)
                input_ = {'input_indices': input_indices, 'target': input['labels'], 'correct_labels': correct_labels}
                output_ = {'target': output['logits'], 'loss': output['loss']}
                # print('outputloss', output['loss'])
                # print('outputlogits', output['logits'])
            else:
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(**input)
                input_ = {'target': input['target']}
                output_ = {'target': output['target'], 'loss': output['loss']}
            metric.add('test', input_, output_)
            evaluation = metric.evaluate('test', 'batch', input_, output_)
            print('evaluation_for_batch', evaluation)
            logger.append(evaluation, 'test', input_size)
            record_pruing_info(model, logger)
            # return
            # break
            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                batch_time = (time.time() - start_time) / (i + 1)
                exp_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                print('running_info', info)
        print('inference_duration', inference_duration)
        evaluation = metric.evaluate('test', 'full')
        print('evaluation_for_full', evaluation)
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
        logger.append(info, 'test')
        print(logger.write('test', metric.metric_name['test']), flush=True)
        model_prof.stop_profile()
        print("Debug 12.2: Test logger created", flush=True)
    return inference_duration


if __name__ == "__main__":
    main()

