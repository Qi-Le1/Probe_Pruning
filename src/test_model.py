import argparse
import os
import time
import torch
import datetime
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate, make_batchnorm_stats
from metric import make_metric, make_logger
from model import make_model, make_prune_model
from module import save, to_device, process_control, resume, makedir_exist_ok
from deepspeed.profiling.flops_profiler import FlopsProfiler

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
NUM_PARAMETER_UNIT = (1000000, 'Million')
FLOPS_UNIT = (1000000, 'Million')
# already in seconds unit
TIME_UNIT = (1, 's')

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
    result_path = os.path.join('output', 'result')
    makedir_exist_ok(result_path)
    dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
    model, tokenizer = make_model(cfg['model_name'])
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    model_prof = FlopsProfiler(model)
    model = model.to(cfg['device'])
    if cfg['model_name'] in ['cnn', 'resnet18', 'wresnet28x2']:
        model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
    # test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    # test(data_loader['test'], model, model_prof, metric, test_logger)
    # vanilla_info_list = get_model_profile('vanilla', model_prof)

    model, tokenizer = make_model(cfg['model_name'])
    # model.load_state_dict(result['model_state_dict'])
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    model = make_prune_model(model, test_logger)
    model_prof = FlopsProfiler(model)
    model = model.to(cfg['device'])
    test(data_loader['test'], model, model_prof, metric, test_logger)
    pruned_info_list = get_model_profile('pruned', model_prof)
    
    batch_num = len(data_loader['test'])
    # summarize_info_list(vanilla_info_list, pruned_info_list, batch_num)
    result = {'cfg': cfg, 'epoch': cfg['epoch'], 'logger_state_dict': {'test': test_logger.state_dict()}}
    save(result, os.path.join(result_path, cfg['model_tag']))
    return

def get_model_profile(tag, model_prof):
    info_list = []
    for name, module in model_prof.model.named_modules():
        temp = [name, module.__flops__, module.__duration__, module.__params__, module.__macs__, type(module)]
        info_list.append(temp)
    return info_list

def summarize_info_list(vanilla_info_list, pruned_info_list, batch_num, logger):

    print('Summary ---------\n')
    vanilla_total_flops = sum([vanilla_info_list[i][1] for i in range(len(vanilla_info_list))])
    pruned_total_flops = sum([pruned_info_list[i][1] for i in range(len(pruned_info_list))])
    print(f"Vanilla FLOPs ({FLOPS_UNIT[1]}): ", vanilla_total_flops/FLOPS_UNIT[0], flush=True)
    print(f"Pruned FLOPs ({FLOPS_UNIT[1]}): ", pruned_total_flops/FLOPS_UNIT[0], flush=True)
    print('Pruning FLOPs reduction percentage (%): ', ((vanilla_total_flops - pruned_total_flops) / vanilla_total_flops) * 100, flush=True)

    vanilla_total_inference_time = sum([vanilla_info_list[i][2] for i in range(len(vanilla_info_list))])
    pruned_total_inference_time = sum([pruned_info_list[i][2] for i in range(len(pruned_info_list))])
    print(f"Vanilla inference time ({TIME_UNIT[1]}): ", vanilla_total_inference_time/TIME_UNIT[0], flush=True)
    print(f"Pruned inference time ({TIME_UNIT[1]}): ", pruned_total_inference_time/TIME_UNIT[0], flush=True)
    print(f"Pruning inference time cost ({TIME_UNIT[1]}): ", (pruned_total_inference_time - vanilla_total_inference_time), flush=True)
    print(f"Pruning inference time cost ({TIME_UNIT[1]}) per batch: ", (pruned_total_inference_time - vanilla_total_inference_time)/(batch_num), flush=True)

    for i in range(len(vanilla_info_list)):
        sub_vanilla_info = vanilla_info_list[i]
        sub_pruned_info = pruned_info_list[i+1]
        print('----\n')
        print(f"VANILLA: {sub_vanilla_info[0]} - {sub_vanilla_info[1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {sub_vanilla_info[2]/TIME_UNIT[0]:.2f} {TIME_UNIT[1]} - {sub_vanilla_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - {sub_vanilla_info[4]}", flush=True)
        print(f"PRUNED : {sub_pruned_info[0]} - {sub_pruned_info[1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {sub_pruned_info[2]/TIME_UNIT[0]:.2f} {TIME_UNIT[1]} - {sub_pruned_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - {sub_pruned_info[4]}", flush=True)
    print('Summary Finished ---------\n')

def test(data_loader, model, model_prof, metric, logger):
    start_time = time.time()
    with torch.no_grad():
        model_prof.start_profile()
        model = model.to(cfg['device'])
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
            logger.append(evaluation, 'test', input_size)
            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                batch_time = (time.time() - start_time) / (i + 1)
                exp_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                print('running_info', info)
        evaluation = metric.evaluate('test', 'full')
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(cfg['model_tag'])]}
        logger.append(info, 'test')
        print(logger.write('test', metric.metric_name['test']))
        model_prof.stop_profile()
    return


if __name__ == "__main__":
    main()
