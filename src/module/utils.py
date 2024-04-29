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
            if 'model.embed_tokens.weight' in name or 'model.norm.weight' in name or 'lm_head.weight' in name:
                continue

            if onlyprobe:
            #     # when only use probe, dont need to calculate for this 2 layers
            #     # just use to match the output shape
            #     if 'down_proj' in name or 'o_proj' in name:
            #         temp = [name, 0, 0, 0, type(module)]
                if not hasattr(module, 'is_pruned') or module.is_pruned == False:
                    temp = [name, 0, 0, 0, type(module)]
        
        if hasattr(module, 'is_pruned') and module.is_pruned == True:
            temp.append(True)

        info_list.append(temp)
    return copy.deepcopy(info_list)

def check_dense_model():
    current_script_dir = os.path.dirname(__file__)
    result_path = os.path.join(current_script_dir, '..', 'output', 'result')
    dense_name_list = cfg['model_tag'].split('_')
    # batch_size
    dense_name_list[4] = str(cfg[cfg['model_name']]['batch_size']['test'])
    # prune_ratio
    dense_name_list[6] = '0'
    # prune_metric
    dense_name_list[7] = 'None'
    # prune_method
    dense_name_list[8] = 'dense'
    # mode
    dense_name_list[9] = 'None'
    # calib_info
    dense_name_list[10] = 'None'
    # prune_info
    dense_name_list[11] = 'None'
    # cust_tgt_modules
    dense_name_list[12] = 'None'
    dense_model_path = os.path.join(result_path, '_'.join(dense_name_list))
    if not os.path.exists(dense_model_path):
        dense_model_path = os.path.join(result_path, 'dense', '_'.join(dense_name_list))
        if not os.path.exists(dense_model_path):
            return None
        else:
            return dense_model_path
    else:
        return dense_model_path
    

def load_dense_model():    
    from .io import load
    dense_model_path = check_dense_model()
    if dense_model_path is None:
        return None, None
    dense_res = load(dense_model_path)
    dense_info_list, dense_duration = dense_res['dense_info_list'], dense_res['dense_duration']
    return dense_info_list, dense_duration
    
def summarize_info_list(pruned_info_list, pruned_duration, logger, dataset_size, onlyprobe_info_list=None):
    # total = fullinf + probe
    # for asyncintra, the info has the dirty write issue because we open 2 streams, check sync mode for the correct info
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
            'dense_duration_token_per_second': dataset_size*cfg['seq_len']/dense_duration,
            'pruned_duration': pruned_duration,
            'pruned_duration_per_sample': pruned_duration/dataset_size,
            'pruned_duration_token_per_second': dataset_size*cfg['seq_len']/pruned_duration,
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
        print(f'dense_duration_token_per_second', dataset_size*cfg['seq_len']/dense_duration, flush=True)
        print(f'pruned_duration_token_per_second', dataset_size*cfg['seq_len']/pruned_duration, flush=True)

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
            'pruned_duration_token_per_second': dataset_size*cfg['seq_len']/pruned_duration,
        }

        print(f"Pruned inference time ({TIME_UNIT[1]}): ", pruned_duration/TIME_UNIT[0], flush=True)
        print(f"Pruned inference time ({TIME_UNIT[1]}) per sample: ", pruned_duration/TIME_UNIT[0]/(dataset_size), flush=True)
        print(f'pruned_duration_token_per_second', dataset_size*cfg['seq_len']/pruned_duration, flush=True)

        print(f"Pruned FLOPs ({FLOPS_UNIT[1]}): ", pruned_total_flops/FLOPS_UNIT[0], flush=True)
        print('Summary Finished ---------\n')
        logger.append(info, 'test')

    return dense_info_list, dense_duration


def model_forward(model, input, inference_duration, index):
    torch.cuda.synchronize(cfg['cuda_default_stream'])
    torch.cuda.nvtx.range_push("iteration{}".format(index))
    start_time = time.time()
    output = model(**input)
    torch.cuda.synchronize(cfg['cuda_default_stream'])
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


# def summarize_fix_probe_info_list(dense_info_list, pruned_info_list, dense_duration, pruned_duration, batch_num, logger):

#     print('Summary ---------\n')
#     dense_total_flops = sum([dense_info_list[i][1] for i in range(len(dense_info_list))])
#     pruned_total_flops = sum([pruned_info_list[i][1] for i in range(len(pruned_info_list))])
#     print(f"dense FLOPs ({FLOPS_UNIT[1]}): ", dense_total_flops/FLOPS_UNIT[0], flush=True)
#     print(f"Pruned FLOPs ({FLOPS_UNIT[1]}): ", pruned_total_flops/FLOPS_UNIT[0], flush=True)
#     print('Pruning FLOPs reduction percentage (%): ', ((dense_total_flops - pruned_total_flops) / (dense_total_flops + 1e-6)) * 100, flush=True)

#     # dense_total_inference_time = sum([dense_info_list[i][2] for i in range(len(dense_info_list))])
#     # pruned_total_inference_time = sum([pruned_info_list[i][2] for i in range(len(pruned_info_list))])
#     print(f"dense inference time ({TIME_UNIT[1]}): ", dense_duration/TIME_UNIT[0], flush=True)
#     print(f"dense inference time ({TIME_UNIT[1]}) per batch: ", dense_duration/TIME_UNIT[0]/batch_num, flush=True)
#     print(f"Pruned inference time ({TIME_UNIT[1]}): ", pruned_duration/TIME_UNIT[0], flush=True)
#     print(f"Pruned inference time ({TIME_UNIT[1]}) per batch: ", pruned_duration/TIME_UNIT[0]/batch_num, flush=True)
#     print(f"Pruning inference time cost ({TIME_UNIT[1]}): ", (pruned_duration - dense_duration), flush=True)
#     print(f"Pruning inference time cost ({TIME_UNIT[1]}) per batch: ", (pruned_duration - dense_duration)/(batch_num), flush=True)

#     info = {
#         'dense_total_FLOPs': dense_total_flops,
#         'Pruned_total_FLOPs': pruned_total_flops,
#         'dense_duration': dense_duration,
#         'dense_duration_per_batch': dense_duration/batch_num,
#         'pruned_duration': pruned_duration,
#         'pruned_duration_per_batch': pruned_duration/batch_num,
#         'pruned_duration_cost_per_batch': (pruned_duration - dense_duration)/(batch_num),
#         'total_FLOPs_ratio': pruned_total_flops/(dense_total_flops+1e-6),
#     }

#     total_target_used_params = 0
#     total_target_params = 0
#     for i in range(len(dense_info_list)):
#         sub_dense_info = dense_info_list[i]
#         sub_pruned_info = pruned_info_list[i+1]
#         if sub_pruned_info[-1] == True:
#             info[f"{sub_pruned_info[-2]}_pruned_FLOPs_ratio"] = sub_pruned_info[1]/(sub_dense_info[1] + 1e-6)
#             total_target_used_params += sub_pruned_info[1]/(sub_dense_info[1] + 1e-6) * sub_pruned_info[3]
#             total_target_params += sub_pruned_info[3]
#         print('----\n')
#         print(f"dense: {sub_dense_info[0]} - {sub_dense_info[1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {sub_dense_info[2]/TIME_UNIT[0]:.2f} {TIME_UNIT[1]} - {sub_dense_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - {sub_dense_info[4]}", flush=True)
#         print(f"PRUNED : {sub_pruned_info[0]} - {sub_pruned_info[1]/FLOPS_UNIT[0]:.2f} {FLOPS_UNIT[1]}Flops - {sub_pruned_info[2]/TIME_UNIT[0]:.2f} {TIME_UNIT[1]} - {sub_pruned_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - {sub_pruned_info[4]}", flush=True)
    
#     if 'unstruct' in cfg['prune_name']:
#         info['FLOPs_for_pruned_layers'] = cfg['prune_ratio']
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
