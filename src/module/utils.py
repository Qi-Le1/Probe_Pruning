import numpy as np
import os
import re
import time
import copy
import torch 
import torch.nn as nn 
from functools import wraps
from config import cfg
import torch
from collections.abc import Iterable, Sequence, Mapping
from itertools import repeat


KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
NUM_PARAMETER_UNIT = (1000000, 'Million')
FLOPS_UNIT = (1000000, 'Million')
# already in seconds unit
TIME_UNIT = (1, 's')

def check_skip_layers(name):
    layer_order_matches = re.findall(r'\d+', name)
    if layer_order_matches:  # Check if the list is not empty
        layer_order = int(layer_order_matches[0])  # Convert the first match to an integer
        if layer_order <= cfg['skip_layers']:
            return True
        return False
    else:
        raise ValueError(f"Layer order not found in the layer key {name}")
    
def get_model_profile(tag, model_prof, onlyprobe=False):
    info_list = []
    for name, module in model_prof.model.model.named_modules():
        temp = [name, module.__flops__, module.__params__, module.__macs__, type(module)]
        # layer_order_matches = re.findall(r'\d+', name)
        # if layer_order_matches:  # Check if the list is not empty
        #     layer_order = int(layer_order_matches[0])  # Convert the first match to an integer
        #     if layer_order <= cfg['skip_layers']:
        #         continue

        if 'llama' in cfg['model_name']: 
            if 'model.embed_tokens' in name or 'model.norm' in name or 'lm_head' in name:
                continue

            if onlyprobe:
                if not hasattr(module, 'is_pruned') or module.is_pruned == False:
                    temp = [name, 0, 0, 0, type(module)]
        elif 'opt' in cfg['model_name']: 
            if 'model.embed_tokens' in name or 'model.norm' in name or 'lm_head' in name:
                continue

            if onlyprobe:
                if not hasattr(module, 'is_pruned') or module.is_pruned == False:
                    temp = [name, 0, 0, 0, type(module)]
        
        # calculate flops ratio only for pruned part
        if hasattr(module, 'is_pruned') and module.is_pruned == True:
            temp.append(True)

        info_list.append(temp)
    return copy.deepcopy(info_list)


    
def summarize_info_list(pruned_info_list, pruned_duration, logger, dataset_size, onlyprobe_info_list=None):
    from .io import load_dense_model
    # total = fullinf + probe
    # for asyncintra, the info has the dirty write issue because we open 2 streams, check sync mode for the correct info
    
    # different package, cannot load, but flops is the same as probe pruning
    if 'llmpruner' in cfg['prune_method'] or 'loraprune' in cfg['prune_method']:
        dense_info_list, dense_duration = None, 0
    else:
        dense_info_list, dense_duration = load_dense_model()
    
    print('Summary ---------\n')
    if dense_info_list is not None and pruned_info_list is not None:
        dense_total_flops = sum([dense_info_list[i][1] for i in range(len(dense_info_list))])
        pruned_total_flops = sum([pruned_info_list[i][1] for i in range(len(pruned_info_list))])
        if onlyprobe_info_list is not None:
            pruned_probe_flops = sum([onlyprobe_info_list[i][1] for i in range(len(onlyprobe_info_list))])
            pruned_fullinf_flops = pruned_total_flops - pruned_probe_flops
        else:
            pruned_probe_flops = 0
            pruned_fullinf_flops = pruned_total_flops

        info = {}
        pruned_layer_dense_total_flops = 0
        pruned_layer_pruned_total_flops = 0
        pruned_layer_pruned_fullinf_flops = 0
        pruned_layer_pruned_probe_flops = 0
        for i in range(len(dense_info_list)):
            sub_dense_info = dense_info_list[i]
            sub_pruned_info = pruned_info_list[i]
            print('----\n')
            if sub_pruned_info[-1] == True:
                info[f"{sub_pruned_info[-2]}_pruned_FLOPs_ratio"] = sub_pruned_info[1]/(sub_dense_info[1] + 1e-6)
                pruned_layer_dense_total_flops += sub_dense_info[1]
                pruned_layer_pruned_total_flops += sub_pruned_info[1]

                pruned_layer_probe_flops = 0
                if onlyprobe_info_list is not None:
                    pruned_layer_probe_flops = onlyprobe_info_list[i][1]
                pruned_layer_pruned_fullinf_flops += sub_pruned_info[1] - pruned_layer_probe_flops
                pruned_layer_pruned_probe_flops += pruned_layer_probe_flops

            print(f"Dense: {sub_dense_info[0]} - {sub_dense_info[1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {sub_dense_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - typemodule: {sub_dense_info[4]}", flush=True)
            print(f"Total after PRUNED : {sub_pruned_info[0]} - {sub_pruned_info[1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {sub_pruned_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - typemodule: {sub_pruned_info[4]}", flush=True)
            print(f"Total after Pruned FLOPs ratio: {sub_pruned_info[1]/(sub_dense_info[1] + 1e-6)}", flush=True)
            if onlyprobe_info_list is not None:
                print(f"Probe after PRUNED : {onlyprobe_info_list[i][0]} - {onlyprobe_info_list[i][1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {onlyprobe_info_list[i][3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - typemodule: {onlyprobe_info_list[i][4]}", flush=True)
                print(f"Probe afterPruned FLOPs ratio: {onlyprobe_info_list[i][1]/(sub_dense_info[1] + 1e-6)}", flush=True)
        
        info = {
            'dense_total_FLOPs': dense_total_flops,
            'Pruned_total_FLOPs': pruned_total_flops,
            'dense_duration': dense_duration,
            'dense_duration_per_sample': dense_duration/dataset_size,
            'dense_duration_token_per_second': dataset_size*cfg['max_seq_len']/dense_duration,
            'pruned_duration': pruned_duration,
            'pruned_duration_per_sample': pruned_duration/dataset_size,
            'pruned_duration_token_per_second': dataset_size*cfg['max_seq_len']/pruned_duration,
            'pruned_duration_cost_per_sample': (pruned_duration - dense_duration)/(dataset_size),
            'total_FLOPs_ratio_for_all_layers': pruned_total_flops / (dense_total_flops + 1e-6),
            'total_FLOPs_ratio_for_pruned_layers': pruned_layer_pruned_total_flops / (pruned_layer_dense_total_flops + 1e-6),
            'fullinf_FLOPs_ratio_for_all_layers': pruned_fullinf_flops / (dense_total_flops + 1e-6),
            'fullinf_FLOPs_ratio_for_pruned_layers': pruned_layer_pruned_fullinf_flops / (pruned_layer_dense_total_flops + 1e-6),
            'probe_FLOPs_ratio_for_all_layers': pruned_probe_flops / (dense_total_flops + 1e-6),
            'probe_FLOPs_ratio_for_pruned_layers': pruned_layer_pruned_probe_flops / (pruned_layer_dense_total_flops + 1e-6),
        }


        print(f"dense inference time ({TIME_UNIT[1]}): ", dense_duration/TIME_UNIT[0], flush=True)
        print(f"dense inference time ({TIME_UNIT[1]}) per sample: ", dense_duration/TIME_UNIT[0]/(dataset_size), flush=True)
        print(f"Pruned inference time ({TIME_UNIT[1]}): ", pruned_duration/TIME_UNIT[0], flush=True)
        print(f"Pruned inference time ({TIME_UNIT[1]}) per sample: ", pruned_duration/TIME_UNIT[0]/(dataset_size), flush=True)
        print(f"Inference time diff ({TIME_UNIT[1]}): ", (pruned_duration - dense_duration), flush=True)
        print(f"Inference time diff ({TIME_UNIT[1]}) per sample: ", (pruned_duration - dense_duration)/(dataset_size), flush=True)
        print(f'dense_duration_token_per_second', dataset_size*cfg['max_seq_len']/dense_duration, flush=True)
        print(f'pruned_duration_token_per_second', dataset_size*cfg['max_seq_len']/pruned_duration, flush=True)

        print(f"dense FLOPs ({FLOPS_UNIT[1]}): ", dense_total_flops/FLOPS_UNIT[0], flush=True)
        print(f"Pruned FLOPs ({FLOPS_UNIT[1]}): ", pruned_total_flops/FLOPS_UNIT[0], flush=True)
        print('total_FLOPs_ratio_for_all_layers: ', info['total_FLOPs_ratio_for_all_layers'], flush=True)
        print("total_FLOPs_ratio_for_pruned_layers", info['total_FLOPs_ratio_for_pruned_layers'])
        print('fullinf_FLOPs_ratio_for_all_layers: ', info['fullinf_FLOPs_ratio_for_all_layers'], flush=True)
        print("fullinf_FLOPs_ratio_for_pruned_layers", info['fullinf_FLOPs_ratio_for_pruned_layers'])
        print('probe_FLOPs_ratio_for_all_layers: ', info['probe_FLOPs_ratio_for_all_layers'], flush=True)
        print("probe_FLOPs_ratio_for_pruned_layers", info['probe_FLOPs_ratio_for_pruned_layers'])
        print('Summary Finished ---------\n')
        logger.append(info, 'test')
    else:
        pruned_total_flops = sum([pruned_info_list[i][1] for i in range(len(pruned_info_list))])
        info = {}
        
        pruned_layer_pruned_total_flops = 0
        for i in range(len(pruned_info_list)):
            sub_pruned_info = pruned_info_list[i]
            print('----\n')
            if sub_pruned_info[-1] == True:
                pruned_layer_pruned_total_flops += sub_pruned_info[1]

            print(f"PRUNED : {sub_pruned_info[0]} - {sub_pruned_info[1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {sub_pruned_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - typemodule: {sub_pruned_info[4]}", flush=True)
        
        info = {
            'Pruned_total_FLOPs': pruned_total_flops,
            'pruned_duration': pruned_duration,
            'pruned_duration_per_sample': pruned_duration/dataset_size,
            'pruned_duration_token_per_second': dataset_size*cfg['max_seq_len']/pruned_duration,
        }

        print(f"Pruned inference time ({TIME_UNIT[1]}): ", pruned_duration/TIME_UNIT[0], flush=True)
        print(f"Pruned inference time ({TIME_UNIT[1]}) per sample: ", pruned_duration/TIME_UNIT[0]/(dataset_size), flush=True)
        print(f'pruned_duration_token_per_second', dataset_size*cfg['max_seq_len']/pruned_duration, flush=True)

        print(f"Pruned FLOPs ({FLOPS_UNIT[1]}): ", pruned_total_flops/FLOPS_UNIT[0], flush=True)
        print('Summary Finished ---------\n')
        logger.append(info, 'test')

    return dense_info_list, dense_duration


def model_forward(model, input, inference_duration, index):
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("iteration{}".format(index))
    start_time = time.time()
    output = model(**input)
    torch.cuda.synchronize()
    cur_inference_duration = time.time() - start_time
    inference_duration += cur_inference_duration
    torch.cuda.nvtx.range_pop()
    print(f'index: {index} - inference_duration: {cur_inference_duration}', flush=True)
    # not considering edge case for time cost
    if index == 0:
        return output, 0
    return output, inference_duration


def nearest_multiple(num_prune, total, multiple):
    remain = (total - num_prune) % multiple
    if remain == 0:
        return num_prune
    else:
        adjusted_prune = num_prune - (multiple - remain)
        return adjusted_prune


def record_pruing_info(model, logger):
    for name, module in model.named_modules():
        if hasattr(module, 'pruning_module'):
            # print('module.pruning_module.pruning_info', module.pruning_module.pruning_info)
            logger.append(module.pruning_module.pruning_info, 'test')
            module.pruning_module.reset_pruning_info()
    return


def update_model_prof(model_prof):
    # https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/profiling/flops_profiler/profiler.py
    # dont need time_hook attacted on every module
    # cause it sync the default stream every time
    for name, module in model_prof.model.named_modules():
        if hasattr(module, "__start_time_hook_handle__"):
            module.__start_time_hook_handle__.remove()
            # del module.__start_time__
            
        if hasattr(module, "__end_time_hook_handle__"):
            module.__end_time_hook_handle__.remove()
            # del module.__duration__
    return 


def match_prefix(model_path):
    # Assume cfg['model_tag'] and model_path are defined
    model_tag_prefix = '_'.join(cfg['model_tag'].split('_')[:3])

    # Find folders matching the prefix
    matching_folders = [folder for folder in os.listdir(model_path) 
                        if os.path.isdir(os.path.join(model_path, folder)) 
                        and folder.startswith(model_tag_prefix)]

    # Process the matching folders
    if matching_folders:
        for folder in matching_folders:
            full_path = os.path.join(model_path, folder)
            return full_path
            # You can add more processing here if needed
    else:
        print("No matching folders found.")

def ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, Sequence):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, Mapping):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output
