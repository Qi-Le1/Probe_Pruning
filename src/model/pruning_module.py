import re
import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import random
import collections
from config import cfg
from module import nearest_multiple
from .hf.utils import mean_process
from scipy import stats
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import os

        
class HiddenRepresentationPruning():

    def __init__(self, cfg, key, modelconfig=None):
        self.key = key
        self.prune_metric = cfg['prune_metric']
        
        if 'gridsearch' in cfg['prune_method']:
            self.attn_prune_ratio = cfg['prune_ratio'][0]
            self.mlp_prune_ratio = cfg['prune_ratio'][1]
        else:
            # if gridratio or flap ratio in prune method, will update later
            self.prune_ratio = self.adjust_prune_ratio(modelconfig)
            
        # self.logger_detailed_info = cfg['logger_detailed_info']

        # self.pq_p = cfg['pq_p']
        # self.pq_q = cfg['pq_q']
        # self.pq_gamma = cfg['pq_gamma']
        # self.pq_beta = cfg['pq_beta']
        # self.eta = self.prune_ratio
        


    def adjust_prune_ratio(self, modelconfig):
        if modelconfig is not None:
            num_hidden_layers = modelconfig.num_hidden_layers
            prune_ratio = num_hidden_layers / (num_hidden_layers - (cfg['skip_layers'] + 1)) * cfg['prune_ratio'] 
            return prune_ratio
        else:
            return cfg['prune_ratio']
    

    def cal_attn_prune_metric(self, probe_out, weight, metric_type, global_metric_score_distribution=None, selected_indices=None):
        probe_num = probe_out.size(0)
        if 'ppwandasp' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])
                global_metric_score_distribution = global_metric_score_distribution.to(probe_out.device)

                # only for seq rank
                if selected_indices is not None and 'seqrank' in cfg['prune_info']:
                    global_metric_score_distribution = global_metric_score_distribution[selected_indices, :]

                if 'probefixratio' in cfg['prune_info']:
                    combined_probe_out = cfg['probefixratio'] * global_metric_score_distribution + (1-cfg['probefixratio']) * norm_probe_out_square
                # dynaratio, since all nonnegative, no need to abs
                else:  
                    denominator = norm_probe_out_square + global_metric_score_distribution

                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight if there is a nan
                    probe_ratio = norm_probe_out_square / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * global_metric_score_distribution + probe_ratio * norm_probe_out_square

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = torch.linalg.vector_norm((combined_probe_out.reshape((1,-1)) * torch.pow(weight, 2)).reshape(4096, 32, 128), ord=2, dim=(0, 2)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            else:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=(0,1)) ** 2 / probe_num, max=cfg['data_type_max'])
                probe_out_dim_metric = torch.linalg.vector_norm((norm_probe_out_square.reshape((1,-1)) * torch.pow(weight, 2)).reshape(4096, 32, 128), ord=2, dim=(0, 2)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
        elif 'wandasp' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])
                global_metric_score_distribution = global_metric_score_distribution.to(probe_out.device)

                # only for seq rank
                if selected_indices is not None and 'seqrank' in cfg['prune_info']:
                    global_metric_score_distribution = global_metric_score_distribution[selected_indices, :]

                if 'probefixratio' in cfg['prune_info']:
                    combined_probe_out = cfg['probefixratio'] * global_metric_score_distribution + (1-cfg['probefixratio']) * norm_probe_out_square
                # dynaratio, since all nonnegative, no need to abs
                else:  
                    denominator = norm_probe_out_square + global_metric_score_distribution

                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight if there is a nan
                    probe_ratio = norm_probe_out_square / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * global_metric_score_distribution + probe_ratio * norm_probe_out_square

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = (torch.sqrt(combined_probe_out.reshape((1,-1))) * torch.abs(weight)).sum(dim=0).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            else:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=(0,1)) ** 2 / probe_num, max=cfg['data_type_max'])
                probe_out_dim_metric = (torch.sqrt(norm_probe_out_square.reshape((1,-1))) * torch.abs(weight)).sum(dim=0).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
        elif 'flap' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                mean_probe_out = torch.mean(probe_out, dim=(0, 1))
                probe_variance = torch.sum(torch.pow(probe_out - mean_probe_out.reshape(1, 1, -1), 2), dim=0).clamp(max=cfg['data_type_max']) / probe_num
                global_metric_score_distribution = global_metric_score_distribution.to(probe_out.device)

                # only for seq rank
                if selected_indices is not None and 'seqrank' in cfg['prune_info']:
                    global_metric_score_distribution = global_metric_score_distribution[selected_indices, :]

                if 'probefixratio' in cfg['prune_info']:
                    combined_probe_out = cfg['probefixratio'] * global_metric_score_distribution + (1-cfg['probefixratio']) * probe_variance
                # dynaratio, weighted by abs
                else:  
                    denominator = probe_variance + global_metric_score_distribution

                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight if there is a nan
                    probe_ratio = probe_variance / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * global_metric_score_distribution + probe_ratio * probe_variance

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = (combined_probe_out * torch.sum(torch.pow(weight, 2), dim=0)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            else:
                mean_probe_out = torch.mean(probe_out, dim=(0, 1), max=cfg['data_type_max'])
                probe_variance = torch.sum(torch.pow(probe_out - mean_probe_out.reshape(1, 1, -1), 2), dim=0).clamp(max=cfg['data_type_max']) / probe_num
                probe_out_dim_metric = (probe_variance * torch.sum(torch.pow(weight, 2), dim=0)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric

    
    def cal_mlp_prune_metric(self, probe_out, weight, metric_type, global_metric_score_distribution=None, selected_indices=None):
        probe_num = probe_out.size(0)
        if 'ppwandasp' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])
                global_metric_score_distribution = global_metric_score_distribution.to(probe_out.device)

                # only for seq rank
                if selected_indices is not None and 'seqrank' in cfg['prune_info']:
                    global_metric_score_distribution = global_metric_score_distribution[selected_indices, :]

                if 'probefixratio' in cfg['prune_info']:
                    combined_probe_out = cfg['probefixratio'] * global_metric_score_distribution + (1-cfg['probefixratio']) * norm_probe_out_square
                # dynaratio, since all nonnegative, no need to abs
                else:  
                    denominator = norm_probe_out_square + global_metric_score_distribution
                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight if there is a nan
                    probe_ratio = norm_probe_out_square / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * global_metric_score_distribution + probe_ratio * norm_probe_out_square

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = torch.linalg.vector_norm((combined_probe_out.reshape((1,-1)) * torch.pow(weight, 2)), ord=2, dim=0).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            else:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=(0,1)) ** 2 / probe_num, max=cfg['data_type_max'])
                probe_out_dim_metric = torch.linalg.vector_norm((norm_probe_out_square.reshape((1,-1)) * torch.pow(weight, 2)), ord=2, dim=0).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
        elif 'wandasp' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])
                global_metric_score_distribution = global_metric_score_distribution.to(probe_out.device)

                # only for seq rank
                if selected_indices is not None and 'seqrank' in cfg['prune_info']:
                    global_metric_score_distribution = global_metric_score_distribution[selected_indices, :]

                if 'probefixratio' in cfg['prune_info']:
                    combined_probe_out = cfg['probefixratio'] * global_metric_score_distribution + (1-cfg['probefixratio']) * norm_probe_out_square
                # dynaratio, since all nonnegative, no need to abs
                else:  
                    denominator = norm_probe_out_square + global_metric_score_distribution

                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight if there is a nan
                    probe_ratio = norm_probe_out_square / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * global_metric_score_distribution + probe_ratio * norm_probe_out_square

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = (torch.sqrt(combined_probe_out.reshape((1,-1))) * torch.abs(weight)).sum(dim=0).clamp(max=cfg['data_type_max'])                    
                return probe_out_dim_metric
            else:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=(0,1)) ** 2 / probe_num, max=cfg['data_type_max'])
                probe_out_dim_metric = (torch.sqrt(norm_probe_out_square.reshape((1,-1))) * torch.abs(weight)).sum(dim=0).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
        elif 'flap' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                mean_probe_out = torch.mean(probe_out, dim=(0, 1))
                probe_variance = torch.sum(torch.pow(probe_out - mean_probe_out.reshape(1, 1, -1), 2), dim=0).clamp(max=cfg['data_type_max']) / probe_num
                global_metric_score_distribution = global_metric_score_distribution.to(probe_out.device)

                # only for seq rank
                if selected_indices is not None and 'seqrank' in cfg['prune_info']:
                    global_metric_score_distribution = global_metric_score_distribution[selected_indices, :]

                if 'probefixratio' in cfg['prune_info']:
                    combined_probe_out = cfg['probefixratio'] * global_metric_score_distribution + (1-cfg['probefixratio']) * probe_variance
                # dynaratio, weighted by abs
                else:  
                    denominator = probe_variance + global_metric_score_distribution
                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight if there is a nan
                    probe_ratio = probe_variance / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * global_metric_score_distribution + probe_ratio * probe_variance

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = (combined_probe_out * torch.sum(torch.pow(weight, 2), dim=0)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            else:
                mean_probe_out = torch.mean(probe_out, dim=(0, 1), max=cfg['data_type_max'])
                probe_variance = torch.sum(torch.pow(probe_out - mean_probe_out.reshape(1, 1, -1), 2), dim=0).clamp(max=cfg['data_type_max']) / probe_num
                probe_out_dim_metric = (probe_variance * torch.sum(torch.pow(weight, 2), dim=0)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            
    def cal_attn_calib_prune_metric(self, calib, weight, metric_type):
        if 'ppwandasp' in metric_type:
            calib = torch.sum(calib, dim=0).clamp(max=cfg['data_type_max'])
            probe_out_dim_metric = torch.linalg.vector_norm((calib.reshape((1,-1)) * torch.pow(weight, 2)).reshape(4096, 32, 128), ord=2, dim=(0, 2)).clamp(max=cfg['data_type_max'])
        elif 'wandasp' in metric_type:
            calib = torch.sum(calib, dim=0).clamp(max=cfg['data_type_max'])
            probe_out_dim_metric = (torch.sqrt(calib).reshape((1,-1)) * torch.abs(weight)).sum(dim=0).clamp(max=cfg['data_type_max'])
        elif 'flap' in metric_type:
            calib = torch.sum(calib, dim=0).clamp(max=cfg['data_type_max'])
            probe_out_dim_metric = (calib * torch.sum(torch.pow(weight, 2), dim=0)).clamp(max=cfg['data_type_max'])

        nan_mask = torch.isnan(probe_out_dim_metric)
        inf_mask = torch.isinf(probe_out_dim_metric)
        print('attn nan_mask', nan_mask)
        print('attn inf_mask', inf_mask)
        if torch.any(nan_mask):
            warnings.warn(f'Found NaN in the attn probe_out_dim_metric, setting to 0')
        if torch.any(inf_mask):
            warnings.warn(f'Found Inf in the attn probe_out_dim_metric, setting to 0')
        sorted_probe_out_dim_metric, indices = torch.sort(probe_out_dim_metric)

        # Print the sorted tensor
        print("sorted_probe_out_dim_metric:", sorted_probe_out_dim_metric, sorted_probe_out_dim_metric.shape)
        return probe_out_dim_metric

    def cal_mlp_calib_prune_metric(self, calib, weight, metric_type):
        if 'ppwandasp' in metric_type:
            calib = torch.sum(calib, dim=0).clamp(max=cfg['data_type_max'])
            probe_out_dim_metric = torch.linalg.vector_norm((calib.reshape((1,-1)) * torch.pow(weight, 2)), ord=2, dim=0).clamp(max=cfg['data_type_max'])
        elif 'wandasp' in metric_type:
            calib = torch.sum(calib, dim=0).clamp(max=cfg['data_type_max'])
            probe_out_dim_metric = (torch.sqrt(calib).reshape((1,-1)) * torch.abs(weight)).sum(dim=0).clamp(max=cfg['data_type_max'])
        elif 'flap' in metric_type:
            calib = torch.sum(calib, dim=0).clamp(max=cfg['data_type_max'])
            probe_out_dim_metric = (calib * torch.sum(torch.pow(weight, 2), dim=0)).clamp(max=cfg['data_type_max'])
        
        nan_mask = torch.isnan(probe_out_dim_metric)
        inf_mask = torch.isinf(probe_out_dim_metric)
        print('mlp nan_mask', nan_mask)
        print('mlp inf_mask', inf_mask)
        if torch.any(nan_mask):
            warnings.warn(f'Found NaN in the mlp probe_out_dim_metric, setting to 0')
        if torch.any(inf_mask):
            warnings.warn(f'Found Inf in the mlp probe_out_dim_metric, setting to 0')
        sorted_probe_out_dim_metric, indices = torch.sort(probe_out_dim_metric)

        # Print the sorted tensor
        print("sorted_probe_out_dim_metric:", sorted_probe_out_dim_metric, sorted_probe_out_dim_metric.shape)
        return probe_out_dim_metric
    
    def sort_attn_metric(self, probe_out_dim_metric, num_heads, head_dim, prune_way, prune_module, multiple, pruning_ratio=None):
        if prune_way == None:
            return None, None, num_heads, head_dim

        if 'gridsearch' in cfg['prune_method']:
            prune_ratio = pruning_ratio if pruning_ratio is not None else self.attn_prune_ratio
        else:
            prune_ratio = pruning_ratio if pruning_ratio is not None else self.prune_ratio
        print('attn prune_ratio', prune_ratio)
        if 'whole' in prune_way:    
            probe_out_dim_metric = probe_out_dim_metric.reshape(num_heads, -1)
            summed_metrics = torch.clamp(probe_out_dim_metric.sum(dim=-1), max=cfg['data_type_max'])
            sorted_value, sorted_indices = torch.sort(summed_metrics, dim=0)
            num_prune_heads = int(prune_ratio * num_heads)
            num_prune_heads = nearest_multiple(head_dim * num_prune_heads, probe_out_dim_metric.numel(), multiple, head_dim) // head_dim
            # Select the heads to prune
            heads_to_preserve = sorted_indices[num_prune_heads:]
            print('num_prune_heads', num_prune_heads)
            full_indices_to_preserve = (torch.arange(head_dim, device=probe_out_dim_metric.device) + heads_to_preserve.unsqueeze(1) * head_dim).view(-1)
            num_heads = num_heads - num_prune_heads
            return full_indices_to_preserve, None, num_heads, head_dim
        else:
            raise NotImplementedError

    def sort_mlp_metric(self, probe_out_dim_metric, multiple, pruning_ratio=None):        
        if 'gridsearch' in cfg['prune_method']:
            prune_ratio = pruning_ratio if pruning_ratio is not None else self.mlp_prune_ratio
        else:
            prune_ratio = pruning_ratio if pruning_ratio is not None else self.prune_ratio
        print('mlp prune_ratio', prune_ratio)
        sorted_value, sorted_indices = torch.sort(probe_out_dim_metric, dim=0)
        num_prune = int(prune_ratio * probe_out_dim_metric.shape[0])
        print('mlp num_prune', num_prune)
        num_prune = nearest_multiple(num_prune, probe_out_dim_metric.shape[0], multiple)
        return sorted_indices[num_prune:], sorted_indices[:num_prune]
    
    def flap_ratio(self, model):
        attn_metric_list, mlp_metric_list = [], []
        standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

        if 'llama' in cfg['model_name']:
            for name, module in model.named_modules():   
                if 'down_proj' not in name and 'o_proj' not in name:
                    continue

                if 'down_proj' in name and 'down_proj' not in cfg['cust_tgt_modules']:
                    continue
                elif 'o_proj' in name and 'o_proj' not in cfg['cust_tgt_modules']:
                    continue

                numbers = int(''.join(filter(str.isdigit, name)))
                if numbers <= cfg['skip']:
                    continue

                if 'o_proj' in name:
                    # flap code: W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                    # we dont put the manually added square (only added for attn) here since it is unreasonable
                    metric = self.cal_attn_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric'])
                    attn_metric_list.append(metric)
                elif 'down_proj' in name:
                    metric = self.cal_mlp_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric'])
                    mlp_metric_list.append(metric)
                
            if len(attn_metric_list) > 0:
                attn_metric = torch.stack(attn_metric_list)
                attn_metric = standarlization(attn_metric)
                attn_metric = attn_metric.reshape(attn_metric.shape[0], -1, 128).mean(dim=2)
            else:
                attn_metric = None

            if len(mlp_metric_list) > 0:
                mlp_metric = torch.stack(mlp_metric_list)
                mlp_metric = standarlization(mlp_metric)
            else:
                mlp_metric = None

            # prune 1 head will lead to 128 times more flops pruned than 1 mlp channel
            multiples = model.config.hidden_size//model.config.num_attention_heads
            if attn_metric is not None and mlp_metric is not None:
                prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
                labels = torch.cat([torch.full_like(attn_metric, multiples).view(-1), torch.full_like(mlp_metric, 1).view(-1)])
            elif attn_metric is not None:
                prune_metric = attn_metric.view(-1)
                labels = torch.full_like(attn_metric, multiples).view(-1)
            elif mlp_metric is not None:
                prune_metric = mlp_metric.view(-1)
                labels = torch.full_like(mlp_metric, 1).view(-1)


            sorted_prune, indices = torch.sort(prune_metric, descending=True)
            sorted_labels = labels[indices]
            # threshold = sorted_prune[int(sorted_prune.numel() * cfg['prune_ratio'])]
            cumulative_sum = torch.cumsum(sorted_labels, dim=0)

            # Calculate the total sum and the target sum
            total_sum = sorted_labels.sum()
            target_sum = total_sum * cfg['prune_ratio']

            # Find the index where the cumulative sum reaches or exceeds the target sum
            threshold_index = torch.searchsorted(cumulative_sum, target_sum, right=True)

            # Set the threshold using the value at this index
            threshold = sorted_prune[threshold_index]

            print("Threshold:", threshold.item())

            for name, module in model.named_modules():
                if 'down_proj' not in name and 'o_proj' not in name:
                    continue
                
                if 'down_proj' in name and 'down_proj' not in cfg['cust_tgt_modules']:
                    continue
                elif 'o_proj' in name and 'o_proj' not in cfg['cust_tgt_modules']:
                    continue

                numbers = int(''.join(filter(str.isdigit, name)))
                print('numebres', numbers)
                if numbers <= cfg['skip_layers']:
                    continue
                
                if 'o_proj' in name:
                    # flap code: W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                    # we dont put the manually added square (only added for attn) here since it is unreasonable
                    metric = self.cal_attn_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric'])
                    metric = metric.reshape(metric.shape[0], -1, 128).mean(dim=2)
                elif 'down_proj' in name:
                    metric = self.cal_mlp_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric'])

                metric = standarlization(metric)
                module.pruning_ratio = metric[metric < threshold].numel() / metric.numel()
                print('name', name, 'module.pruning_ratio', module.pruning_ratio)
        elif 'opt' in cfg['model_name']:
            pass
        return

    def grid_ratio(self, model):
        from module import TRANSFORMERS_MODELS_TO_GRID_SEARCH_RATIO
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
                if numbers <= cfg['skip_layers']:
                    continue
                
                if 'o_proj' in name:
                    # if there are more layers, like llama-2-13B, prune less in each layer to reach mean prune ratio
                    # simply scale this ratio, one may run the grid search for other model type to find the best ratio
                    pruning_ratio = TRANSFORMERS_MODELS_TO_GRID_SEARCH_RATIO['llama']['128'][cfg['prune_ratio']]['o_proj'] * \
                        (32 / 29) / (model.config.num_hidden_layers / (model.config.num_hidden_layers - (cfg['skip_layers'] + 1)))
                    module.pruning_ratio = pruning_ratio
                elif 'down_proj' in name:
                    pruning_ratio = TRANSFORMERS_MODELS_TO_GRID_SEARCH_RATIO['llama']['128'][cfg['prune_ratio']]['down_proj'] * \
                        (32 / 29) / (model.config.num_hidden_layers / (model.config.num_hidden_layers - (cfg['skip_layers'] + 1)))
                    module.pruning_ratio = pruning_ratio


        elif 'opt' in cfg['model_name']:
            pass
        return


    # def grid_search(self, model):
    #     if 'llama' in cfg['model_name']:
    #         for name, module in model.named_modules():
    #             if 'down_proj' not in name and 'o_proj' not in name:
    #                 continue
                
    #             if 'down_proj' in name and 'down_proj' not in cfg['cust_tgt_modules']:
    #                 continue
    #             elif 'o_proj' in name and 'o_proj' not in cfg['cust_tgt_modules']:
    #                 continue

    #             numbers = int(''.join(filter(str.isdigit, name)))
    #             print('numebers', numbers)
    #             if numbers <= cfg['skip_layers']:
    #                 continue
                
    #             if 'o_proj' in name:
    #                 module.pruning_ratio = cfg['prune_ratio'][0]
    #             elif 'down_proj' in name:
    #                 module.pruning_ratio = cfg['prune_ratio'][1]


    #     elif 'opt' in cfg['model_name']:
    #         pass
    #     return