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
    record_pruing_info, get_model_profile, summarize_info_list, match_prefix, load, update_model_prof, model_forward, remove_non_picklable_items
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

def prepare_cude_events(model):
    if 'llama-2' in cfg['model_name']:
        for i in range(model.config.num_hidden_layers):
            cfg[f'cuda_events_mlp_{i}'] = torch.cuda.Event()
        

def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    result_path = os.path.join('output', 'result')
    makedir_exist_ok(result_path)

    cfg['epoch'] = 0 
    dense_name_list = cfg['model_tag'].split('_')
    # batch_size
    dense_name_list[4] = str(cfg[cfg['model_name']]['batch_size']['test'])
    # prune_hyper
    dense_name_list[6] = '0'
    # prune_metric
    dense_name_list[7] = 'None'
    # prune_method
    dense_name_list[8] = 'dense'
    # mode
    dense_name_list[9] = 'sync'
    # calib_info
    dense_name_list[10] = 'None'
    # probe_type
    dense_name_list[11] = 'None'
    # cust_tgt_modules
    dense_name_list[12] = 'None'
    dense_model_path = os.path.join(result_path, '_'.join(dense_name_list))
    if not os.path.exists(dense_model_path):
        dense_model_path = os.path.join(result_path, 'dense', '_'.join(dense_name_list))
    dense_res = load(dense_model_path)
    dense_info_list, dense_duration = dense_res['dense_info_list'], dense_res['dense_duration']

    dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
    model, tokenizer = make_model(cfg['model_name'])
    prepare_cude_events(model)
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
        print('len(calibration_data_loader)', len(calibration_data_loader['train']))
        run_calibration(model, calibration_data_loader['train'])
        cfg['calibration_stage'] = False
        print('Calibration Done...')
    model_prof = FlopsProfiler(model)
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    # return
    inference_duration = test(data_loader['test'], model, model_prof, metric, test_logger)
    pruned_info_list = get_model_profile('pruned', model_prof)
    
    # print('dense_info_list', dense_info_list[0], dense_info_list[1])
    summarize_info_list(dense_info_list, pruned_info_list, dense_duration, inference_duration, test_logger)
    evaluation = metric.evaluate('test', 'full')
    print('evaluation_for_full', evaluation)
    # thread lock bug
    test_logger.writer = None
    remove_non_picklable_items(cfg)
    result = {'cfg': cfg, 'epoch': cfg['epoch'], 'logger': {'test': test_logger},\
              'dense_info_list': dense_info_list, 'pruned_info_list': pruned_info_list, \
              'dense_duration': dense_duration, 'pruned_duration': inference_duration, 'dataset_size': cfg['dataset_size']['test']}
    save(result, os.path.join(result_path, cfg['model_tag']))
    return

def cal_prune_metric(x, weight, metric_type):
    if 'savemetricseq' in cfg['prune_method']:
        x = torch.clamp(torch.sum(x, dim=0), min=cfg['data_type_min'], max=cfg['data_type_max'])
        # x = torch.clamp(torch.norm(x, p=2, dim=0) ** 2, min=cfg['data_type_min'], max=cfg['data_type_max'])
    if 'wandasp' in metric_type:
        # probe_out_dim_metric = (torch.sqrt(norm_squared.unsqueeze_(0).reshape((1,-1))) * torch.abs(weight)).sum(dim=0)
        pass
    elif 'flap' in metric_type:
        pass
    elif 'probe' in metric_type:    
        x = torch.sqrt(((x.unsqueeze(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(min=cfg['data_type_min'], max=cfg['data_type_max']))
    return x

def global_determine_ratio(model, if_log=False):
    attn_metric_list, mlp_metric_list = [], []
    if 'llama' in cfg['model_name']:
        for name, module in model.named_modules():   
            if 'down_proj' not in name and 'o_proj' not in name:
                continue

            if 'down_proj' in name and 'down_proj' not in cfg['cust_tgt_modules']:
                continue
            elif 'o_proj' in name and 'o_proj' not in cfg['cust_tgt_modules']:
                continue

            numbers = int(''.join(filter(str.isdigit, name)))
            print('numebres', numbers)
            if numbers <= cfg['skip']:
                continue
            x = copy.deepcopy(module.get_global_metric_score_distribution())
            x = cal_prune_metric(x, module.weight.data, cfg['prune_metric'])
            if if_log:
                x = torch.log(x + 1)
            # if 'savemetricseq' in cfg['prune_method']:
            #     x = torch.clamp(torch.sum(x, dim=0), min=cfg['data_type_min'], max=cfg['data_type_max'])
            if 'globalratiostd' in cfg['prune_method']:
                if 'down_proj' in name:
                    x = (x - torch.mean(x, axis=-1, keepdim=True)) / torch.std(x, axis=-1, keepdim=True)
                    mlp_metric_list.append(x)
                elif 'o_proj' in name:
                    if 'each' in cfg['vo_prune_way']:
                        num_heads = model.config.num_attention_heads
                        head_dim = model.config.hidden_size // num_heads
                        x_reshaped = x.view(num_heads, head_dim)
                        mean = torch.mean(x_reshaped, axis=-1, keepdim=True)
                        std = torch.std(x_reshaped, axis=-1, keepdim=True)
                        x_normalized = (x_reshaped - mean) / std
                        x = x_normalized.view(1, model.config.hidden_size)
                    else:
                        x = (x - torch.mean(x, axis=-1, keepdim=True)) / torch.std(x, axis=-1, keepdim=True)
                    attn_metric_list.append(x)
            # Print the module name and attribute name
            print('name', name)
        
        if len(attn_metric_list) > 0:
            attn_metric = torch.stack(attn_metric_list)
            sorted_prune, indices = torch.sort(attn_metric.view(-1))
            attn_threshold = sorted_prune[int(sorted_prune.numel() * cfg['prune_hyper'])]
        if len(mlp_metric_list) > 0:
            mlp_metric = torch.stack(mlp_metric_list)
            sorted_prune, indices = torch.sort(mlp_metric.view(-1))
            mlp_threshold = sorted_prune[int(sorted_prune.numel() * cfg['prune_hyper'])]

        output_dir = "output/vis"
        os.makedirs(output_dir, exist_ok=True)

        for name, module in model.named_modules():
            if 'down_proj' not in name and 'o_proj' not in name:
                continue
            
            if 'down_proj' in name and 'down_proj' not in cfg['cust_tgt_modules']:
                continue
            elif 'o_proj' in name and 'o_proj' not in cfg['cust_tgt_modules']:
                continue

            numbers = int(''.join(filter(str.isdigit, name)))
            print('numebres', numbers)
            if numbers <= cfg['skip']:
                continue
            x = copy.deepcopy(module.get_global_metric_score_distribution())
            x = cal_prune_metric(x, module.weight.data, cfg['prune_metric'])
            if if_log:
                x = torch.log(x + 1)
            # if 'savemetricseq' in cfg['prune_method']:
            #     x = torch.clamp(torch.sum(x, dim=0), min=cfg['data_type_min'], max=cfg['data_type_max'])
            print('name', name, 'x', x, 'sorted_x', torch.sort(x)[0])
            print('mean', torch.mean(x, axis=-1), 'std', torch.std(x, axis=-1))
            if 'globalratiostd' in cfg['prune_method']:
                x = (x - torch.mean(x, axis=-1, keepdim=True)) / torch.std(x, axis=-1, keepdim=True)
            sorted_metric, indices = torch.sort(x)

            # Your existing code to plot sorted_metric
            # plt.figure()
            # plt.plot(sorted_metric.cpu().numpy())
            # plt.title(f'Sorted Metric for {name}')
            # plt.xlabel('Index')
            # plt.ylabel('Metric Value')
            # plt.savefig(os.path.join(output_dir, f"{name.replace('/', '_')}_sorted_metric.png"))
            # plt.close()

            # New code to plot and save the PDF using seaborn's kdeplot
            # plt.figure()
            # sns.kdeplot(sorted_metric.cpu().numpy(), bw_adjust=0.5)
            # plt.title(f'PDF of Sorted Metric for {name}')
            # plt.xlabel('Metric Value')
            # plt.ylabel('Density')
            # pdf_filename = f"{name.replace('/', '_')}_sorted_metric_pdf.png"
            # plt.savefig(os.path.join(output_dir, pdf_filename))
            # plt.close()
            
            # data = sorted_metric.cpu().numpy()
            # # Plot the CDF using seaborn's kdeplot with the cumulative option
            # plt.figure()
            # sns.kdeplot(data, bw_adjust=0.5, cumulative=True, fill=True)
            # plt.title(f'CDF of Sorted Metric for {name}')
            # plt.xlabel('Metric Value')
            # plt.ylabel('CDF')

            # # Save the plot
            # cdf_filename = f"{name.replace('/', '_')}_sorted_metric_cdf.png"
            # plt.savefig(os.path.join(output_dir, cdf_filename))
            # plt.close()
            # # # Print statement for the new PDF plot
            # # print(f"PDF plot saved to {os.path.join(output_dir, pdf_filename)}")
            
            if 'down_proj' in name and len(mlp_metric_list) > 0:
                module.pruning_ratio = sorted_metric[sorted_metric < mlp_threshold].numel() / sorted_metric.numel()
                print('threshold', mlp_threshold, 'module.sorted_metric', sorted_metric)
                print('name', name, 'module.pruning_ratio', module.pruning_ratio)
            elif 'o_proj' in name and len(attn_metric_list) > 0:
                module.pruning_ratio = sorted_metric[sorted_metric < attn_threshold].numel() / sorted_metric.numel()
            # module.pruning_ratio = sorted_metric[sorted_metric < threshold].numel() / sorted_metric.numel()
            
                print('threshold', attn_threshold, 'module.sorted_metric', sorted_metric)
                print('name', name, 'module.pruning_ratio', module.pruning_ratio)
        
    elif 'opt' in cfg['model_name']:
        pass

    
    return

def run_calibration(model, data_loader):
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
            # elif cfg['task_name'] in ['csr']:
            #     input_size = input['labels'].size(0)
            #     input_indices = input['input_indices']
            #     correct_labels = input['correct_labels']
            #     # print('input', input)
            #     input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
            #             'labels': input['labels']}
            #     input = to_device(input, cfg['device'])
            #     output = model(**input)
            #     input_ = {'input_indices': input_indices, 'target': input['labels'], 'correct_labels': correct_labels}
            #     output_ = {'target': output['logits'], 'loss': output['loss']}
            #     # print('outputloss', output['loss'])
            # else:
            #     input = collate(input)
            #     input_size = input['data'].size(0)
            #     input = to_device(input, cfg['device'])
            #     output = model(**input)
            #     input_ = {'target': input['target']}
            #     output_ = {'target': output['target'], 'loss': output['loss']}
            # if cfg['task_name'] == 's2s':
            #     output_['generate'] = model.generate(input_ids=input["input_ids"],
            #                                         max_new_tokens=cfg['max_new_tokens'])
            # elif cfg['task_name'] == 'clm':
            #     if cfg['data_name'] in ['dolly']:
            #         output_['generate'] = model.generate(input_ids=input["input_ids"],
            #                                             attention_mask=input["attention_mask"],
            #                                             max_new_tokens=cfg['max_new_tokens'],
            #                                             eos_token_id=cfg['pad_token_id'],
            #                                             no_repeat_ngram_size=2)
            # break
            if iterate_small_samples:
                if i == 100:
                    break
        if 'globalratio' in cfg['prune_method']:
            global_determine_ratio(model)
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

        data_loader_iter = iter(data_loader)
        input = next(data_loader_iter)
        cfg['cur_batch_index'] += 1
        # warm up pytorch
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
        update_model_prof(model_prof)
        torch.cuda.cudart().cudaProfilerStart()
        for i, input in enumerate(data_loader):
            cfg['cur_batch_index'] += 1
            torch.cuda.nvtx.range_push("iteration{}".format(i))
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

            metric.add('test', input_, output_)
            evaluation = metric.evaluate('test', 'batch', input_, output_)
            print('evaluation_for_batch', evaluation)
            logger.append(evaluation, 'test', input_size)
            record_pruing_info(model, logger)
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
            torch.cuda.nvtx.range_pop()
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
        evaluation = metric.evaluate('test', 'full')
        print('evaluation_for_full', evaluation)
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
        logger.append(info, 'test')
        print(logger.write('test', metric.metric_name['test']), flush=True)
        model_prof.stop_profile()

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

