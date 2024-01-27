import numpy as np
import os
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

def nearest_even_number(value):
    rounded_value = round(value)
    # If it's odd, adjust by 1 to make it even
    return rounded_value if rounded_value % 2 == 0 else rounded_value + 1

def identity_function(x):
    return x

def alternate_broadcast(tensor1, tensor2):
    if tensor1.device != tensor2.device:
        # Move tensor2 to the device of tensor1
        tensor2 = tensor2.to(tensor1.device)
    tensor1 = tensor1.sum(dim=0)
    # Calculate the total number of dimensions after broadcasting
    return tensor1 * tensor2

def record_pruing_info(model, logger):
    for name, module in model.named_modules():
        if hasattr(module, 'pruning_module'):
            # print('module.pruning_module.pruning_info', module.pruning_module.pruning_info)
            logger.append(module.pruning_module.pruning_info, 'test')
            module.pruning_module.reset_pruning_info()
    return

def get_model_profile(tag, model_prof):
    info_list = []
    for name, module in model_prof.model.named_modules():
        temp = [name, module.__flops__, module.__duration__, module.__params__, module.__macs__, type(module)]
        # print('temp', temp)
        if hasattr(module, 'is_pruned'):
            print('module.key', module.key)
            temp.append(module.key)
            temp.append(True)
        info_list.append(temp)
    
    def get_module_duration(module):
        duration = module.__duration__
        if hasattr(module, 'pruning_module'):
            duration -= module.pruning_module.logger_info_time_used
        if duration == 0:  # e.g. ModuleList
            for m in module.children():
                duration += get_module_duration(m)
        return duration

    duration = get_module_duration(model_prof.model)
    # print('duration', duration, type(duration))
    return copy.deepcopy(info_list), duration


def summarize_info_list(vanilla_info_list, pruned_info_list, vanilla_duration, pruned_duration, batch_num, logger):

    print('Summary ---------\n')
    vanilla_total_flops = sum([vanilla_info_list[i][1] for i in range(len(vanilla_info_list))])
    pruned_total_flops = sum([pruned_info_list[i][1] for i in range(len(pruned_info_list))])
    
    info = {
        'vanilla_total_FLOPs': vanilla_total_flops,
        'Pruned_total_FLOPs': pruned_total_flops,
        'vanilla_duration': vanilla_duration,
        'vanilla_duration_per_batch': vanilla_duration/batch_num,
        'pruned_duration': pruned_duration,
        'pruned_duration_per_batch': pruned_duration/batch_num,
        'pruned_duration_cost_per_batch': (pruned_duration - vanilla_duration)/(batch_num),
        'total_FLOPs_ratio': pruned_total_flops/(vanilla_total_flops+1e-6),
    }
    total_target_used_params = 0
    total_target_params = 0
    for i in range(len(vanilla_info_list)):
        sub_vanilla_info = vanilla_info_list[i]
        sub_pruned_info = pruned_info_list[i+1]
        if sub_pruned_info[-1] == True:
            info[f"{sub_pruned_info[-2]}_pruned_FLOPs_ratio"] = sub_pruned_info[1]/(sub_vanilla_info[1] + 1e-6)
            # [name, module.__flops__, module.__duration__, module.__params__, module.__macs__, type(module)]
            print('sub_pruned_info', sub_pruned_info)
            print('sub_vanilla_info', sub_vanilla_info, sub_pruned_info[1]/(sub_vanilla_info[1] + 1e-6))
            total_target_used_params += sub_pruned_info[1]/(sub_vanilla_info[1] + 1e-6) * sub_vanilla_info[3]
            total_target_params += sub_vanilla_info[3]
        print('----\n')
        print(f"VANILLA: {sub_vanilla_info[0]} - {sub_vanilla_info[1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {sub_vanilla_info[2]/TIME_UNIT[0]:.2f} {TIME_UNIT[1]} - {sub_vanilla_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - {sub_vanilla_info[4]}", flush=True)
        print(f"PRUNED : {sub_pruned_info[0]} - {sub_pruned_info[1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {sub_pruned_info[2]/TIME_UNIT[0]:.2f} {TIME_UNIT[1]} - {sub_pruned_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - {sub_pruned_info[4]}", flush=True)
    
    if 'unstruct' in cfg['prune_name']:
        info['FLOPs_for_pruned_layers'] = cfg['prune_hyper']
    else:
        info['FLOPs_for_pruned_layers'] = total_target_used_params / (total_target_params + 1e-6)
    
    

    # vanilla_total_inference_time = sum([vanilla_info_list[i][2] for i in range(len(vanilla_info_list))])
    # pruned_total_inference_time = sum([pruned_info_list[i][2] for i in range(len(pruned_info_list))])
    print(f"Vanilla inference time ({TIME_UNIT[1]}): ", vanilla_duration/TIME_UNIT[0], flush=True)
    print(f"Vanilla inference time ({TIME_UNIT[1]}) per batch: ", vanilla_duration/TIME_UNIT[0]/batch_num, flush=True)
    print(f"Pruned inference time ({TIME_UNIT[1]}): ", pruned_duration/TIME_UNIT[0], flush=True)
    print(f"Pruned inference time ({TIME_UNIT[1]}) per batch: ", pruned_duration/TIME_UNIT[0]/batch_num, flush=True)
    print(f"Pruning inference time cost ({TIME_UNIT[1]}): ", (pruned_duration - vanilla_duration), flush=True)
    print(f"Pruning inference time cost ({TIME_UNIT[1]}) per batch: ", (pruned_duration - vanilla_duration)/(batch_num), flush=True)

    print(f"Vanilla FLOPs ({FLOPS_UNIT[1]}): ", vanilla_total_flops/FLOPS_UNIT[0], flush=True)
    print(f"Pruned FLOPs ({FLOPS_UNIT[1]}): ", pruned_total_flops/FLOPS_UNIT[0], flush=True)
    print('Pruning FLOPs for all modules: ', (pruned_total_flops / (vanilla_total_flops + 1e-6)), flush=True)
    print("info[FLOPs_for_pruned_layers]", info['FLOPs_for_pruned_layers'])
    print('Summary Finished ---------\n')
    logger.append(info, 'test')
    logger.save(False)
    return


# def get_fix_prune_model_profile(tag, model_prof):

#     info_list = []
#     for name, module in model_prof.model.named_modules():
#         temp = [name, module.__flops__, module.__duration__, module.__params__, module.__macs__, type(module)]
#         # print('temp', temp)
#         if hasattr(module, 'pruning_module') or hasattr(module, 'prune_metric'):
#             temp.append(module.key)
#             temp.append(True)
#         info_list.append(temp)
    
#     def get_module_duration(module):
#         duration = module.__duration__
#         if hasattr(module, 'pruning_module'):
#             duration -= module.pruning_module.logger_info_time_used
#         if duration == 0:  # e.g. ModuleList
#             for m in module.children():
#                 duration += get_module_duration(m)
#         return duration

#     duration = get_module_duration(model_prof.model)
#     # print('duration', duration, type(duration))
#     return copy.deepcopy(info_list), duration


# def summarize_fix_prune_info_list(vanilla_info_list, pruned_info_list, vanilla_duration, pruned_duration, batch_num, logger):

#     print('Summary ---------\n')
#     vanilla_total_flops = sum([vanilla_info_list[i][1] for i in range(len(vanilla_info_list))])
#     pruned_total_flops = sum([pruned_info_list[i][1] for i in range(len(pruned_info_list))])
#     print(f"Vanilla FLOPs ({FLOPS_UNIT[1]}): ", vanilla_total_flops/FLOPS_UNIT[0], flush=True)
#     print(f"Pruned FLOPs ({FLOPS_UNIT[1]}): ", pruned_total_flops/FLOPS_UNIT[0], flush=True)
#     print('Pruning FLOPs reduction percentage (%): ', ((vanilla_total_flops - pruned_total_flops) / (vanilla_total_flops + 1e-6)) * 100, flush=True)

#     # vanilla_total_inference_time = sum([vanilla_info_list[i][2] for i in range(len(vanilla_info_list))])
#     # pruned_total_inference_time = sum([pruned_info_list[i][2] for i in range(len(pruned_info_list))])
#     print(f"Vanilla inference time ({TIME_UNIT[1]}): ", vanilla_duration/TIME_UNIT[0], flush=True)
#     print(f"Vanilla inference time ({TIME_UNIT[1]}) per batch: ", vanilla_duration/TIME_UNIT[0]/batch_num, flush=True)
#     print(f"Pruned inference time ({TIME_UNIT[1]}): ", pruned_duration/TIME_UNIT[0], flush=True)
#     print(f"Pruned inference time ({TIME_UNIT[1]}) per batch: ", pruned_duration/TIME_UNIT[0]/batch_num, flush=True)
#     print(f"Pruning inference time cost ({TIME_UNIT[1]}): ", (pruned_duration - vanilla_duration), flush=True)
#     print(f"Pruning inference time cost ({TIME_UNIT[1]}) per batch: ", (pruned_duration - vanilla_duration)/(batch_num), flush=True)

#     info = {
#         'vanilla_total_FLOPs': vanilla_total_flops,
#         'Pruned_total_FLOPs': pruned_total_flops,
#         'vanilla_duration': vanilla_duration,
#         'vanilla_duration_per_batch': vanilla_duration/batch_num,
#         'pruned_duration': pruned_duration,
#         'pruned_duration_per_batch': pruned_duration/batch_num,
#         'pruned_duration_cost_per_batch': (pruned_duration - vanilla_duration)/(batch_num),
#         'total_FLOPs_ratio': pruned_total_flops/(vanilla_total_flops+1e-6),
#     }

#     total_target_used_params = 0
#     total_target_params = 0
#     for i in range(len(vanilla_info_list)):
#         sub_vanilla_info = vanilla_info_list[i]
#         sub_pruned_info = pruned_info_list[i+1]
#         if sub_pruned_info[-1] == True:
#             info[f"{sub_pruned_info[-2]}_pruned_FLOPs_ratio"] = sub_pruned_info[1]/(sub_vanilla_info[1] + 1e-6)
#             total_target_used_params += sub_pruned_info[1]/(sub_vanilla_info[1] + 1e-6) * sub_pruned_info[3]
#             total_target_params += sub_pruned_info[3]
#         print('----\n')
#         print(f"VANILLA: {sub_vanilla_info[0]} - {sub_vanilla_info[1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {sub_vanilla_info[2]/TIME_UNIT[0]:.2f} {TIME_UNIT[1]} - {sub_vanilla_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - {sub_vanilla_info[4]}", flush=True)
#         print(f"PRUNED : {sub_pruned_info[0]} - {sub_pruned_info[1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {sub_pruned_info[2]/TIME_UNIT[0]:.2f} {TIME_UNIT[1]} - {sub_pruned_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - {sub_pruned_info[4]}", flush=True)
    
#     if 'unstruct' in cfg['prune_name']:
#         info['FLOPs_for_pruned_layers'] = cfg['prune_hyper']
#     else:
#         info['FLOPs_for_pruned_layers'] = total_target_used_params / (total_target_params + 1e-6)
    
#     print("info[FLOPs_for_pruned_layers]", info['FLOPs_for_pruned_layers'])
#     print('Summary Finished ---------\n')
#     logger.append(info, 'test')
#     logger.save(False)
#     return


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
