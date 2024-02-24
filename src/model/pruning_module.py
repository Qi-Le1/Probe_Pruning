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
from module import nearest_even_number

class BasePruning:
    def __init__(self, cfg):
        self.prune_name = cfg['prune_name']
        self.prune_metric = cfg['prune_metric']
        self.prune_hyper = cfg['prune_hyper'] 
        self.prune_dim = cfg['prune_dim'] 
        self.logger_detailed_info = cfg['logger_detailed_info']

        self.pq_p = cfg['pq_p']
        self.pq_q = cfg['pq_q']
        self.pq_gamma = cfg['pq_gamma']
        self.pq_beta = cfg['pq_beta']
        self.eta = self.prune_hyper

        self.bin_edges = [
            -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, # -1000 to -100
            -90, -80, -70, -60, -50, -40, -30, -20, -10,  
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100,  # -100 to 100 
            200, 300, 400, 500, 600, 700, 800, 900, 1000  # 100 to 1000
        ]
        fine_bins = np.arange(-10, 10, 0.1).tolist()
        self.bin_edges = self.bin_edges + fine_bins + [1e-3]
        self.bin_edges = sorted(set(self.bin_edges))
        self.reset_pruning_info()

        self.logger_info_time_used = 0
        self.weight_norm_across_channel_dims = None

    def monitor_time(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            args[0].logger_info_time_used += time.time() - start_time
            return result
        return wrapper
    
    def reset_pruning_info(self):
        self.pruning_info = {}
        pass
    
    @monitor_time
    def update_pruning_info(self, info):
        self.pruning_info.update(info)
        pass
    
    @monitor_time
    def cal_repr_distribution(self, h, info_key):
        if not self.logger_detailed_info:
            return
        temp_h = h.detach().to(torch.float32).cpu().numpy()
        hist, _ = np.histogram(temp_h, bins=self.bin_edges)
        self.update_pruning_info({info_key: (hist/temp_h.shape[0]).tolist()})

    def cal_weight_norm_across_channel_dims(self, weight):
        if weight.dim() == 2:
            self.weight_norm_across_channel_dims = torch.linalg.vector_norm(weight, ord=1, dim=0)
        elif weight.dim() == 4:
            self.weight_norm_across_channel_dims = torch.linalg.vector_norm(weight, ord=1, dim=(0, 2, 3))
        else:
            raise ValueError('Not valid weight dimension')
        return


def parallel_cal_varying_length_norm(sorted_norm, norm):
    # sorted_norm is non-negative
    processed_channels = sorted_norm.pow(norm)
    # print('processed_channels', processed_channels.shape, processed_channels[0])
    varying_vector_norm = torch.pow(processed_channels.cumsum(dim=-1), 1/norm)
    return varying_vector_norm
            
def parallel_cal_varying_length_info(sorted_norm, pq_p, pq_q, reversed=False):
    if reversed:
        sorted_norm = torch.flip(sorted_norm, [1])
    nominator_varying_vector_norm = parallel_cal_varying_length_norm(sorted_norm, pq_p)
    denominator_varying_vector_norm = parallel_cal_varying_length_norm(sorted_norm, pq_q)

    nominator_varying_vector_norm = nominator_varying_vector_norm.to(cfg['device'])
    denominator_varying_vector_norm = denominator_varying_vector_norm.to(cfg['device'])
    # print('nominator_varying_vector_norm', nominator_varying_vector_norm.shape, nominator_varying_vector_norm[0])
    # print('denominator_varying_vector_norm', denominator_varying_vector_norm.shape, denominator_varying_vector_norm[0])

    # num_rows, num_cols = nominator_varying_vector_norm.shape

    # if reversed:
    #     # Create a tensor where each row starts from 1 and decreases to the length of the row
    #     dimension = torch.arange(num_cols, 0, -1).unsqueeze(0)
    # else:
        # Create a tensor where each row starts from 1 and increases to the length of the row
    if nominator_varying_vector_norm.dim() == 1:
        dimension = torch.arange(1, nominator_varying_vector_norm.shape[0] + 1).to(cfg['device'])
    else:
        dimension = torch.arange(1, nominator_varying_vector_norm.shape[1] + 1).to(cfg['device'])
        dimension = dimension.expand(nominator_varying_vector_norm.shape[0], -1).to(cfg['device'])
    # dimension = torch.arange(1, nominator_varying_vector_norm.shape[0] + 1).to(cfg['device'])
    # dimension = dimension.expand(nominator_varying_vector_norm.shape[0], -1).to(cfg['device'])
    return nominator_varying_vector_norm, denominator_varying_vector_norm, dimension

def cal_prune_count_base_on_pq(sorted_tensor, pq_p, pq_q, eta, pq_beta, pq_gamma, key):
    
    # norm_across_other_dims = norm_across_other_dims + (norm_across_other_dims == 0) * 1e-9
    # Calculate norms only for non-zero channels
    # non_zero_norms = norm_across_other_dims[non_zero_mask]

    # norm_p = torch.linalg.vector_norm(sorted_tensor, ord=pq_p, dim=0)
    # norm_q = torch.linalg.vector_norm(sorted_tensor, ord=pq_q, dim=0) + 1e-10
    
    # dimension = sorted_tensor.shape[0]
    # pq_indices = (1 - dimension ** (1/pq_q - 1/pq_p) * (norm_p / norm_q))
    
    # # add additional dimension if dimension is 0
    # # if pq_indices.dim() == 0 or pq_indices.dim() == 1:
    # #     pq_indices.unsqueeze_(0)
    # print('pq_indices', pq_indices, dimension)
    # if torch.isnan(pq_indices).any():
    #     pq_indices = torch.min(pq_indices, torch.ones_like(pq_indices))
    #     raise ValueError('pq_indices contains nan values')

    # lower_bound = dimension * (1 + eta) ** (-pq_q / (pq_q - pq_p)) * ((1 - pq_indices) ** (pq_q * pq_p / (pq_q - pq_p)))
    # print('lower_bound', lower_bound, dimension)
    # beta_tensor = torch.full_like(lower_bound, pq_beta)
    # prune_channels_count = torch.floor(dimension * torch.min(pq_gamma * (1 - lower_bound / dimension), beta_tensor))
    # print('prune_channels_count', prune_channels_count)

    # return int(prune_channels_count), pq_indices

    nominator_varying_vector_norm, denominator_varying_vector_norm, dimension = parallel_cal_varying_length_info(sorted_tensor, pq_p, pq_q)
    # print('nominator_varying_vector_norm', nominator_varying_vector_norm.shape, nominator_varying_vector_norm)
    # print('denominator_varying_vector_norm', denominator_varying_vector_norm.shape, denominator_varying_vector_norm)
    pq_indices_varying_length = (1 - dimension ** (1/pq_q - 1/pq_p) * (nominator_varying_vector_norm / denominator_varying_vector_norm))
    lower_bound = dimension * (1 + eta) ** (-pq_q / (pq_q - pq_p)) * ((1 - pq_indices_varying_length) ** (pq_q * pq_p / (pq_q - pq_p)))
    
    # lower_bound = lower_bound.cpu().numpy()
    # x = list(range(len(lower_bound.tolist())))
    # dx = np.diff(x)
    # dy = np.diff(lower_bound)

    # # Compute slope
    # slopes = dy / dx
    
    # if 'low' in cfg['prune_name']:
    #     # avoid edge case of slope
    #     window_size = 21  # 10 neighbors on each side + the element itself

    #     # Create a window with equal weights
    #     window = np.ones(window_size) / window_size
    #     # Calculate the moving average using convolution
    #     averages = np.convolve(slopes, window, 'same')
    #     abs_averages_slopes = np.abs(averages)
    #     # Find the index of the minimum value in abs_slopes
    #     first_phase_transition = np.argmin(abs_averages_slopes)
    #     pq_indices = pq_indices_varying_length[first_phase_transition]
    #     lower_bound = lower_bound[first_phase_transition]
    x = torch.arange(len(lower_bound), dtype=torch.float32, device=lower_bound.device)

    # Calculate differences (equivalent to np.diff)
    dx = x[1:] - x[:-1]
    dy = lower_bound[1:] - lower_bound[:-1]

    # Compute slope
    slopes = dy / dx

    if 'low' in cfg['prune_name']:
        # Avoid edge case of slope, just randomly pick this number
        window_size = 20  # 10 neighbors on each side + the element itself

        # Create a window with equal weights
        window = torch.ones(window_size, dtype=torch.float32,  device=lower_bound.device) / window_size
        window = window.to(lower_bound.device)  # Ensure window is on the same device as lower_bound

        # Calculate the moving average using convolution
        # PyTorch's conv1d expects a 3D tensor (batch, channel, length), so we need to add extra dimensions
        slopes = slopes.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        window = window.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Use conv1d for moving average
        averages = torch.nn.functional.conv1d(slopes, window, padding=window_size//2)
        averages = averages.squeeze()  # Remove extra dimensions

        # negative_values = averages[averages <= 0]
        sign_change_mask = ((torch.sign(averages[:-1]) * torch.sign(averages[1:])) < 0) & (torch.sign(averages[:-1]) > 0)

        # Find indices where sign changes from positive to negative
        # positive_to_negative_mask = (averages[:-1] > 0) & (averages[1:] <= 0)
        transition_indices = torch.where(sign_change_mask)[0] + 1

        if len(transition_indices) > 0:
            first_phase_transition = transition_indices[-1].item()  # Get the last transition point
        else:
            first_phase_transition = lower_bound.shape[0] - 1
            print("No transition from positive to negative found in slopes.")

        # Check if there are any negative values
        # if len(negative_values) > 0:
        #     # Find the maximum among the negative values (closest to zero)
        #     closest_negative = torch.max(negative_values)

        #     # Get the index of this value in the original 'averages' tensor
        #     first_phase_transition = torch.where(averages == closest_negative)[0][0]
        # else:
        #     first_phase_transition = None  # or handle the case where there are no negative values
        #     raise ValueError('No negative values found in averages')

        # print('lower_bound', lower_bound)
        # print('pq_indices_varying_length', pq_indices_varying_length)
        # print('dimension', dimension)
        pq_indices = pq_indices_varying_length[first_phase_transition]
        lower_bound = lower_bound[first_phase_transition]
        dimension = dimension[first_phase_transition]

        print("Index of negative value closest to zero low:", first_phase_transition, sorted_tensor.shape[0], lower_bound, dimension)
    elif 'high' in cfg['prune_name']:
        slopes = torch.abs(dy / dx)
        threshold = 0.05 * slopes.shape[0]
        indices = torch.where(slopes > threshold)[0]
        if len(indices) > 0:
            second_phase_transition = indices[0].item()  # Get the first index as a Python scalar
        else:
            print('dont find second phase transition')
            second_phase_transition = lower_bound.shape[0] - 1  # or handle the case where there are no negative values

        pq_indices = pq_indices_varying_length[second_phase_transition]
        lower_bound = lower_bound[second_phase_transition]
        dimension = dimension[second_phase_transition]
        print("Index of negative value closest to zero high:", second_phase_transition, sorted_tensor.shape[0], lower_bound)
    else:
        pq_indices = pq_indices_varying_length[-1]
        lower_bound = lower_bound[-1]
        dimension = dimension[-1]
        print("Index of negative value closest to zero no:", pq_indices, dimension, sorted_tensor.shape[0], lower_bound)

    beta_tensor = torch.full_like(lower_bound, pq_beta)
    prune_channels_count = torch.floor(dimension * torch.min(pq_gamma * (1 - lower_bound / dimension), beta_tensor))
    print(f'{key}_prune_channels_count_ratio: ', prune_channels_count/sorted_tensor.shape[0])
    prune_channels_count = prune_channels_count.to(cfg['device'])
    pq_indices = pq_indices_varying_length.to(cfg['device'])
    return int(prune_channels_count), pq_indices

def cal_prune_metric(probe_out, weight, metric_type):
    if 'wandasp' in metric_type:
        if probe_out.dim() == 2:
            probe_out.unsqueeze_(0)
        size = probe_out.shape[0]
        weight_factor = 1.0 / size
        sum_squared_norms = torch.sum(torch.norm(probe_out, p=2, dim=1) ** 2 * weight_factor, dim=0)
        average_squared_norm = sum_squared_norms / torch.tensor(size, device=probe_out.device, dtype=torch.float)
        probe_out_dim_metric = (torch.sqrt(average_squared_norm.unsqueeze_(0).reshape((1,-1))) * torch.abs(weight)).sum(dim=0)
    elif 'flap' in metric_type:
        pass
    elif 'probe' in metric_type:
        if probe_out.dim() == 2:
            probe_out.unsqueeze_(0)
        size = probe_out.shape[0]
        weight_factor = 1.0 / size
        average_squared_norm = torch.sum(torch.norm(probe_out, p=2, dim=1) ** 2 * weight_factor, dim=0).clamp(min=None, max=65504)
        probe_out_dim_metric = torch.sqrt(((average_squared_norm.unsqueeze_(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(min=None, max=65504))

    return probe_out_dim_metric

def cal_prune_metric_only_input(probe_out, metric_type):
    if 'wandasp' in metric_type:
        if probe_out.dim() == 2:
            probe_out.unsqueeze_(0)
        size = probe_out.shape[0]
        weight_factor = 1.0 / size
        sum_squared_norms = torch.sum(torch.norm(probe_out, p=2, dim=1) ** 2 * weight_factor, dim=0)
        average_squared_norm = sum_squared_norms / torch.tensor(size, device=probe_out.device, dtype=torch.float)
    elif 'flap' in metric_type:
        pass
    elif 'probe' in metric_type:
        if probe_out.dim() == 2:
            probe_out.unsqueeze_(0)
        size = probe_out.shape[0]
        weight_factor = 1.0 / size
        average_squared_norm = torch.sum(torch.norm(probe_out, p=2, dim=1) ** 2 * weight_factor, dim=0).clamp(min=None, max=65504)
    return average_squared_norm

def cal_running_mean_prune_metric(running_mean, weight, metric_type):
    if 'wandasp' in metric_type:
        probe_out_dim_metric = (torch.sqrt(running_mean.reshape((1,-1))) * torch.abs(weight)).sum(dim=0)
    elif 'flap' in metric_type:
        pass
    elif 'probe' in metric_type:
        probe_out_dim_metric = torch.sqrt(((running_mean.reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(min=None, max=65504))
    return probe_out_dim_metric

class HiddenRepresentationPruning(BasePruning):

    def __init__(self, cfg, key, device=None, in_dim=None, out_dim=None):
        BasePruning.__init__(self, cfg)
        self.key = key
        self.device = device
        if out_dim:
            self.scaler_in = torch.zeros((in_dim), device=self.device)
        self.nsamples = 0
        if 'pq' in cfg['prune_name']:
            self.pq_p = cfg['pq_p']
            self.pq_q = cfg['pq_q']
            self.eta = cfg['prune_hyper']
            self.pq_beta = cfg['pq_beta']
            self.pq_gamma = cfg['pq_gamma']

    def sort_probe_mlp_metric(self, probe_out_dim_metric, multiple):
        probe_out_dim_metric.abs_()
        # probe_out_dim_metric = probe_out_dim_metric.to(torch.float32)
        # mask = torch.ones(probe.shape[-1], dtype=torch.bool, device=probe.device)
        sorted_value, sorted_indices = torch.sort(probe_out_dim_metric, dim=0)
        print(f'{self.key} sorted_value', sorted_value)
        # normalized_sorted_value = sorted_value / sorted_value.sum()
        # print(f'{self.key} normalized_sorted_value', normalized_sorted_value)
        # mean = torch.mean(sorted_value)
        # std = torch.std(sorted_value)

        # # Then, normalize the tensor: (sorted_value - mean) / std
        # standardlized_value = (sorted_value - mean) / std
        # print(f'{self.key} standardlized_value', standardlized_value)
        if 'mag' in cfg['prune_name']:
            num_prune = int(self.prune_hyper * probe_out_dim_metric.shape[0])
        elif 'pq' in cfg['prune_name']:
            num_prune = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma, self.key)[0]
        
        # let the remaining element be the multiple of multiple to fit tensor cores
        num_prune = num_prune + ((probe_out_dim_metric.shape[0] - num_prune) % multiple)
        return sorted_indices[num_prune:], sorted_indices[:num_prune]
    
    # def cal_probe_attn_weights_metric(self, attn_weights_metric):
        
    #     attn_weights_metric = attn_weights_metric.to(torch.float32)
    #     sorted_value, sorted_indices = torch.sort(attn_weights_metric, dim=1)
    #     if 'mag' in cfg['prune_name']:
    #         num_prune = int(self.prune_hyper * attn_weights_metric.shape[1])
    #     # Select indices to prune for each head
    #     indices_to_preserve = sorted_indices[:, num_prune:]

    #     return indices_to_preserve

        # attn_weights_metric.abs_()
        # attn_weights_metric = attn_weights_metric.to(torch.float32)
        # # mask = torch.ones(probe.shape[-1], dtype=torch.bool, device=probe.device)
        # sorted_value, sorted_indices = torch.sort(attn_weights_metric, dim=0)
        # if 'mag' in cfg['prune_name']:
        #     num_prune = int(self.prune_hyper * attn_weights_metric.shape[0])
        # elif 'pq' in cfg['prune_name']:
        #     num_prune = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma)[0]
        #     # print('num_prune', num_prune)
        # # print('prune_channels_count', prune_channels_count, norm_across_other_dims[0].shape[0], self.prune_hyper)

        # return sorted_indices[num_prune:]

    def cal_probe_attn_metric(self, probe_out_dim_metric, num_heads, head_dim, prune_way, prune_module, multiple):
        # mask = torch.ones(probe.shape[-1], dtype=torch.bool, device=probe.device)
        probe_out_dim_metric.abs_()
        probe_out_dim_metric = probe_out_dim_metric.to(torch.float32)
        
        # delete whole head
        if prune_way == 'whole':    
            probe_out_dim_metric = probe_out_dim_metric.reshape(num_heads, -1)
            # Sum over the last dimension and take absolute values
            summed_metrics = torch.abs(probe_out_dim_metric.sum(dim=-1))
            # Sort the summed metrics
            sorted_value, sorted_indices = torch.sort(summed_metrics, dim=0)
            # print('summed_metricssorted_value', sorted_value)
            # Determine the number of heads to prune
            if 'mag' in cfg['prune_name']:
                num_prune_heads = int(num_heads * cfg['prune_hyper'])
            elif 'pq' in cfg['prune_name']:
                num_prune_heads = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma, f'{self.key}_{prune_module}')[0]
            # Select the heads to prune
            heads_to_preserve = sorted_indices[num_prune_heads:]

            head_indices = (torch.arange(head_dim, device=probe_out_dim_metric.device) + heads_to_preserve.unsqueeze(1) * head_dim).view(-1)

            num_heads = num_heads - num_prune_heads

            return head_indices, None, num_heads, head_dim
        elif prune_way == 'each':
            probe_out_dim_metric = probe_out_dim_metric.reshape(num_heads, -1)
            # Sort the probe_out_dim_metric across each head
            sorted_value, sorted_indices = torch.sort(probe_out_dim_metric, dim=1)

            # Select indices to prune for each head
            if 'mag' in cfg['prune_name']:
                # Determine the number of elements to prune in each head
                # handling the edge case for RoPE if prune each head for qk
                num_prune_head_dim = nearest_even_number(probe_out_dim_metric.shape[1] * cfg['prune_hyper'])
                indices_to_preserve = sorted_indices[:, num_prune_head_dim:]
            elif 'pq' in cfg['prune_name']:
                num_prune = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma, f'{self.key}_{prune_module}')[0]

            # Generate a range tensor for head indices
            head_range = torch.arange(num_heads, device=probe_out_dim_metric.device) * head_dim

            # Create the full indices for pruning using broadcasting
            full_indices_to_preserve = (indices_to_preserve + head_range.unsqueeze(1)).view(-1)

            head_dim = head_dim - num_prune_head_dim

            return full_indices_to_preserve, indices_to_preserve, num_heads, head_dim
        elif prune_way == 'fill':

            sorted_value, sorted_indices = torch.sort(probe_out_dim_metric, dim=0)

            print(f'{self.key}_{prune_module} sorted_value', sorted_value)
            # normalized_sorted_value = sorted_value / sorted_value.sum()
            # print(f'{self.key}_{prune_module} normalized_sorted_value', normalized_sorted_value)
            # mean = torch.mean(sorted_value)
            # std = torch.std(sorted_value)

            # # Then, normalize the tensor: (sorted_value - mean) / std
            # standardlized_value = (sorted_value - mean) / std
            # print(f'{self.key}_{prune_module} standardlized_value', standardlized_value)
            # print('sorted_value', sorted_value)
            if 'mag' in cfg['prune_name']:
                num_prune = int(probe_out_dim_metric.shape[0] * cfg['prune_hyper'])
            elif 'pq' in cfg['prune_name']:
                num_prune = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma, f'{self.key}_{prune_module}')[0]
            # print('delete indices', sorted_indices[:num_prune])
            return sorted_indices[num_prune:], None, num_heads, head_dim

            
    def batch_pruning(self, h, layer_type, layer_info, key, is_prune_out_dim):
        # Exclude the first dimension (batch size) and the prune_dim
        # calculate the pq-index per sample
        if h.dim() != 3 and layer_type == 'linear':
            raise ValueError('Not valid input dimension for batch_pruning')
        elif h.dim() != 4 and layer_type == 'conv2d':
            raise ValueError('Not valid input dimension for batch_pruning')

        with torch.no_grad():
            h_shape = h.shape
            h_type = h.dtype
            self.cal_repr_distribution(h, f'{self.key}_dense_hist')
            prune_dim = (h.dim() + self.prune_dim) % h.dim()
            if layer_type == 'linear' and prune_dim != h.dim() - 1:
                raise ValueError('Not valid prune dim')
            elif layer_type == 'conv2d' and prune_dim != 1:
                raise ValueError('Not valid prune dim')

            preserve_channels = self.apply_pruning(h, key, layer_info, prune_dim, is_prune_out_dim)
            # print('preserve_channels2', preserve_channels)
            pruned_h = h.index_select(dim=prune_dim, index=preserve_channels.to(h.device))
            
            return pruned_h, prune_dim, preserve_channels
            
            

    def apply_pruning(self, h, key, layer_info, prune_dim, is_prune_out_dim):
        if prune_dim >= h.dim():
            raise ValueError('Not valid pruning dimension')

        if 'pq' in self.prune_name:
            preserve_channels = self.pq_struct(h, key, layer_info, prune_dim)
            if self.prune_hyper == 9999:
                return [torch.empty(0)]
            return preserve_channels
        elif 'mag' in self.prune_name:
            if self.prune_hyper == 9999:
                return [torch.empty(0)]
            preserve_channels = self.mag_struct(h, key, layer_info, prune_dim, is_prune_out_dim)
            return preserve_channels
        else:
            raise ValueError('Not valid pruning method')
        

    def pq_struct(self, h, key, layer_info, prune_dim):
        info = {}
        
        dims_to_aggregate = tuple(i for i in range(h.dim()) if i != prune_dim)
        if self.prune_metric == 'IF1N':
            norm_across_other_dims = torch.linalg.vector_norm(h, ord=1, dim=dims_to_aggregate)
        elif self.prune_metric == 'IF2N':
            norm_across_other_dims = torch.linalg.vector_norm(h, ord=2, dim=dims_to_aggregate)    

        norm_across_other_dims = norm_across_other_dims + (norm_across_other_dims == 0) * 1e-9
        # print('norm_across_other_dims', norm_across_other_dims, norm_across_other_dims.shape, norm_across_other_dims.dim())
        # Calculate norms only for non-zero channels
        # non_zero_norms = norm_across_other_dims[non_zero_mask]
        norm_p = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_p, dim=1)
        norm_q = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_q, dim=1) + 1e-10
        # print(self.exclude_dim_to_aggregate, dims_to_aggregate, 1, norm_p, norm_q)
        dimension = norm_across_other_dims[0].shape[0]
        pq_indices = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * (norm_p / norm_q))

        if torch.isnan(pq_indices).any():
            raise ValueError('pq_indices contains nan values')

        lower_bound = dimension * (1 + self.eta) ** (-self.pq_q / (self.pq_q - self.pq_p)) * (1 - pq_indices) ** (self.pq_q * self.pq_p / (self.pq_q - self.pq_p))
        beta_tensor = torch.full_like(lower_bound, self.pq_beta)
        prune_channels_count = torch.floor(dimension * torch.min(self.pq_gamma * (1 - lower_bound / dimension), beta_tensor))
        # print('lower_bound', lower_bound.shape, lower_bound)
        # print('pq_indices', pq_indices.shape, pq_indices)
        sorted_norm, sorted_channels = torch.sort(norm_across_other_dims, dim=1)

        eta_zero_lower_bound = dimension * (1 + 0) ** (-self.pq_q / (self.pq_q - self.pq_p)) * (1 - pq_indices) ** (self.pq_q * self.pq_p / (self.pq_q - self.pq_p))
        eta_zero_lower_bound = torch.floor(dimension * self.pq_gamma * (1 - eta_zero_lower_bound / dimension))
        # print('sorted_norm', sorted_norm.shape, sorted_norm)
        if self.prune_hyper == 9999 and cfg['batch_size'] == 1:
            # print('here', self.prune_hyper)
            start_time = time.time()
            
            nominator_varying_vector_norm, denominator_varying_vector_norm, dimension = parallel_cal_varying_length_info(sorted_norm)
            # print('dimension', dimension.shape, dimension)
            pq_indices_varying_length = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * (nominator_varying_vector_norm / denominator_varying_vector_norm))
            
            reversed_nominator_varying_vector_norm, reversed_denominator_varying_vector_norm, dimension = parallel_cal_varying_length_info(sorted_norm, reversed=True)

            reversed_pq_indices_varying_length = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * (reversed_nominator_varying_vector_norm / reversed_denominator_varying_vector_norm))
            
            # print('nominator_varying_vector_norm', nominator_varying_vector_norm.shape, nominator_varying_vector_norm[0])
            # print('denominator_varying_vector_norm', denominator_varying_vector_norm.shape, denominator_varying_vector_norm[0])
            
            # exclude length 1 vector
            pq_indices_varying_length_cut = pq_indices_varying_length[:, 1:pq_indices_varying_length.shape[1]-2]
            # flip back, so it is the result counting from right to left
            reversed_pq_indices_varying_length_cut = torch.flip(reversed_pq_indices_varying_length, [1])[:, 2:reversed_pq_indices_varying_length.shape[1]-1]
            
            nominator_varying_vector_norm_cut = nominator_varying_vector_norm[:, 1:nominator_varying_vector_norm.shape[1]-2]
            reversed_nominator_varying_vector_norm_cut = torch.flip(reversed_nominator_varying_vector_norm, [1])[:, 2:reversed_nominator_varying_vector_norm.shape[1]-1]

            denominator_varying_vector_norm_cut = denominator_varying_vector_norm[:, 1:denominator_varying_vector_norm.shape[1]-2]
            reversed_denominator_varying_vector_norm_cut = torch.flip(reversed_denominator_varying_vector_norm, [1])[:, 2:reversed_denominator_varying_vector_norm.shape[1]-1]

            pq_indices_ratio = pq_indices_varying_length_cut / reversed_pq_indices_varying_length_cut
            p_norm_ratio = nominator_varying_vector_norm_cut / reversed_nominator_varying_vector_norm_cut
            q_norm_ratio = denominator_varying_vector_norm_cut / reversed_denominator_varying_vector_norm_cut
            pq_indices_varying_length = pq_indices_varying_length.mean(dim=0).tolist()
            
            # print('lower_bound', eta_zero_lower_bound[0][0])
            info[f"{key}_pq_lower_bound"] = eta_zero_lower_bound[0][0].item()
            info[f"{key}_pq_indices_varying_lengths"] = pq_indices_varying_length
            reversed_pq_indices_varying_length = reversed_pq_indices_varying_length.mean(dim=0).tolist()
            info[f"{key}_reversed_pq_indices_varying_lengths"] = reversed_pq_indices_varying_length
            info[f"{key}_pq_indices_ratio"] = pq_indices_ratio.mean(dim=0).tolist()
            info[f"{key}_p_norm_ratio"] = p_norm_ratio.mean(dim=0).tolist()
            info[f"{key}_r_trend"] = q_norm_ratio.mean(dim=0).tolist()
            info[f"{key}_q_norm_ratio"] = q_norm_ratio.mean(dim=0).tolist()
            
            self.logger_info_time_used += time.time() - start_time

        preserve_channels = sorted_channels[prune_channels_count.item():]
        logger_norm_across_other_dims = norm_across_other_dims.mean(dim=0).squeeze(0).tolist()

        start_time = time.time()
        info[f"{key}_norm_across_other_dims"] = logger_norm_across_other_dims
        info[f"{key}_pq_indices"] = pq_indices.mean(dim=0).squeeze(0).tolist()
        self.logger_info_time_used += time.time() - start_time
        self.update_pruning_info(info)
        return preserve_channels


    def mag_struct(self, h, key,layer_info, prune_dim, is_prune_out_dim):
        bsz = h.size(0)
        info = {}
        # dims_to_aggregate = tuple(i for i in range(h.dim()) if i != prune_dim)
        if 'wandasp' in self.prune_metric:
            
            #first piece
            sum_squared_norms = torch.sum(torch.norm(h, p=2, dim=1) ** 2, dim=0)

            average_squared_norm = sum_squared_norms / torch.tensor(bsz, device=h.device, dtype=torch.float)

            # Now compute norm_across_other_dims using scaler_inp
            # Assuming layer_info['weight'] is defined and has the appropriate shape
            norm_across_other_dims = (torch.sqrt(average_squared_norm.unsqueeze_(0).reshape((1,-1))) * torch.abs(layer_info['weight'])).sum(dim=0)

            # second piece
            # nsamples = 0
            # for i in range(bsz):
            #     scaler_inp *= nsamples / (nsamples + 1)
            #     scaler_inp += torch.norm(h[i], p=2, dim=1) ** 2 / (nsamples + 1)
            #     nsamples += 1
            #     # mean and sum are the same after sorting
            # norm_across_other_dims = (torch.sqrt(scaler_inp.reshape((1,-1))) * torch.abs(layer_info['weight'])).sum(dim=0)
        elif 'flap' in self.prune_metric:
            # old_baseline_inp = torch.zeros((h.size(1)), device=h.device)
            # self.baseline_inp = torch.mean(h, dim=1) 
            # self.fluc_inp = torch.sum((inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1) / (self.nsamples + batch_size)  
            
            mean_h = torch.mean(h, dim=1)
            norm_across_other_dims = torch.sum((x - mean_x) * torch.linalg.norm(layer_info['weight'], ord=2, dim=0), dim=0) / bsz

        elif 'testourmetric' in self.prune_metric:
            # pass
            sum_squared_norms = torch.sum(torch.norm(h, p=2, dim=1) ** 2, dim=0)

            average_squared_norm = sum_squared_norms / torch.tensor(bsz, device=h.device, dtype=torch.float)
            norm_across_other_dims = torch.sqrt(((average_squared_norm.unsqueeze_(0).reshape((1,-1))) * torch.pow(layer_info['weight'], 2)).sum(dim=0))
            # print('norm_across_other_dims', norm_across_other_dims)

        prune_channels_count = int(self.prune_hyper * norm_across_other_dims.shape[0])
        # print('prune_channels_count', prune_channels_count, norm_across_other_dims[0].shape[0], self.prune_hyper)
        sorted_values, sorted_channels = torch.sort(norm_across_other_dims, dim=0)
        
        preserve_channels = sorted_channels[prune_channels_count:]
        # print('preserve_channels', preserve_channels)
        self.update_pruning_info(info)
        return preserve_channels











    # def batch_pruning(self, h, layer_type, layer_info, key, is_prune_out_dim):
    #     # Exclude the first dimension (batch size) and the prune_dim
    #     # calculate the pq-index per sample
    #     if h.dim() != 3 and layer_type == 'linear':
    #         raise ValueError('Not valid input dimension for batch_pruning')
    #     elif h.dim() != 4 and layer_type == 'conv2d':
    #         raise ValueError('Not valid input dimension for batch_pruning')

    #     with torch.no_grad():
    #         prune_channels_multi_dims = [None] * h.dim()
    #         saving_flops_multi_dims = [0] * h.dim()
    #         h_shape = h.shape
    #         h_type = h.dtype
            
    #         self.cal_repr_distribution(h, f'{self.key}_dense_hist')
    #         if 'unstruct' in self.prune_name:
    #             if self.batch_integ in ['inter', 'union']:
    #                 raise ValueError('Not valid batch integration method')
    #             if 'magunstruct' in self.prune_name:
    #                 flattened_h = h.view(h.size(0), -1)
    #                 if self.prune_metric == 'IF1N':
    #                     norm_along_dim_1 = torch.linalg.vector_norm(flattened_h, ord=1, dim=1)
    #                 elif self.prune_metric == 'IF2N':
    #                     norm_along_dim_1 = torch.linalg.vector_norm(flattened_h, ord=2, dim=1)
    #                 _, sorted_indices = torch.sort(norm_along_dim_1, dim=1)
    #                 num_indices_to_prune = int(self.prune_hyper * sorted_indices.size(1))
    #                 # Select the indices to prune (lowest norms)
    #                 prune_indices = sorted_indices[:, :num_indices_to_prune]
    #                 mask = torch.ones_like(h, dtype=torch.bool)
    #                 if prune_indices is not None:
    #                     # Mark the indices to be pruned as False
    #                     mask[:, prune_indices] = False
    #                 pruned_h = h * mask.to(h.device)
    #                 return pruned_h, None, None
    #             elif 'pqunstruct' in self.prune_name:
    #                 pass
                
    #             elif 'wandaunstruct' in self.prune_name:
    #                 if layer_type == 'linear':
    #                     dim = (0, 1)
    #                     h_norm = torch.linalg.vector_norm(h, ord=2, dim=dim)
    #                     h_norm = h_norm.view(1, -1)
    #                 elif layer_type == 'conv2d':
    #                     raise ValueError('Not valid layer type conv2D')
    #                     # dim = (0, 2, 3)
    #                     # h_norm = torch.linalg.vector_norm(h, ord=2, dim=dim)
    #                     # h_norm = h_norm.view(1, -1, 1, 1)
    #                 metric = layer_info['weight'].abs() * h_norm
    #                 _, sorted_idx = torch.sort(metric, dim=1) 
    #                 pruned_idx = sorted_idx[:,:int(layer_info['weight'].size(1) * self.prune_hyper)] 
    #                 # layer_info['weight'].scatter_(dim=1, index=pruned_idx, src=0)
    #                 # pruned_dims
    #                 # prune_channels_multi_dims
    #                 return h, 'wandaunstrcut', pruned_idx.to(h.device)
    #         elif 'struct' in self.prune_name:
    #             if self.prune_dim_select_mode == 'max':
    #                 for prune_dim in self.prune_dim:
    #                     prune_dim = (h.dim() + prune_dim) % h.dim()
    #                     if layer_type == 'linear' and prune_dim != h.dim() - 1:
    #                         raise ValueError('Not valid prune dim')
    #                     elif layer_type == 'conv2d' and prune_dim != 1:
    #                         raise ValueError('Not valid prune dim')
    #                     prune_channels = self.apply_pruning(h, key, layer_info, prune_dim, is_prune_out_dim)
    #                     prune_channels = self.apply_batch_integ(h_shape[prune_dim], prune_channels)
    #                     prune_channels_multi_dims[prune_dim] = prune_channels

    #                     # saving_flops = self.cal_saving_flops(h, prune_dim, prune_channels, layer_type, layer_info)
    #                     # saving_flops_multi_dims[prune_dim] = saving_flops

    #                 if len(self.prune_dim) == 1:
    #                     final_prune_dim = (h.dim() + self.prune_dim[0]) % h.dim()
    #                 else:
    #                     raise ValueError('Not valid prune dim')
    #                 # final_prune_dim = np.argmax(saving_flops_multi_dims)
    #                 # print('saving_flops_multi_dims', saving_flops_multi_dims)
    #                 # print('prune_channels_multi_dims', prune_channels_multi_dims)
    #                 # final_prune_dim = 2
    #                 pruned_h = self.prune_h(h, final_prune_dim, prune_channels_multi_dims[final_prune_dim])

    #                 # print('final_prune_dim', final_prune_dim)
    #                 # print('saving_flops_multi_dims', saving_flops_multi_dims)
    #                 # print('prune_channels_multi_dims', prune_channels_multi_dims)

    #                 pruned_dims = [dim if dim == final_prune_dim else None for dim in range(h.dim())]
    #                 for dim in range(len(prune_channels_multi_dims)):
    #                     if dim != final_prune_dim:
    #                         prune_channels_multi_dims[dim] = None
    #                         # saving_flops_multi_dims[dim] = 0
                    
    #                 # TODO: hardcode
    #                 if prune_channels_multi_dims[pruned_dims[final_prune_dim]] == None:
    #                     num_pruned_channels = 0
    #                 else:
    #                     num_pruned_channels = prune_channels_multi_dims[pruned_dims[final_prune_dim]].size(-1)

    #                 start_time = time.time()
    #                 cur_batch_info = {
    #                     f"{key}_pruned_dims": pruned_dims[final_prune_dim],
    #                     # f"{key}_pruned_channels": list(prune_channels_multi_dims[pruned_dims[final_prune_dim]]),
    #                     f"{key}_total_channels": h_shape[pruned_dims[final_prune_dim]],
    #                     f"{key}_pruned_ratio": num_pruned_channels / (h_shape[pruned_dims[final_prune_dim]] + 1e-10),
    #                 }
    #                 self.logger_info_time_used += time.time() - start_time
    #                 self.update_pruning_info(cur_batch_info)
    #                 # print('after\n')
    #                 # print('final_prune_dim', final_prune_dim)
    #                 # print('saving_flops_multi_dims', saving_flops_multi_dims)
    #                 # print('prune_channels_multi_dims', prune_channels_multi_dims)
    #                 return pruned_h, pruned_dims, prune_channels_multi_dims
    #         else:
    #             raise ValueError('Not valid pruning method')
            
    #         torch.cuda.empty_cache()

    # def prune_h(self, h, prune_dim, prune_channels):
    #     # if 'probe' in cfg['prune_name']:
    #     #     return h
    #     if 'WOF1N' in cfg['prune_metric'] or 'WOF2N' in cfg['prune_metric']:
    #         return h
    #     # Create a boolean mask for all indices
    #     mask = torch.ones(h.size(prune_dim), dtype=torch.bool)
    #     # print('pre_mask', mask)
    #     # Mark the indices to be pruned as False
    #     if prune_channels is not None:
    #         mask[prune_channels] = False
    #     # print('mask', mask, prune_channels)
    #     # Use the mask to index the tensor
    #     pruned_h = h.index_select(dim=prune_dim, index=mask.nonzero().squeeze().to(h.device))
    #     return pruned_h

    
    
    # def apply_batch_integ(self, cur_total_channels, prune_channels):
    #     if prune_channels[0].numel() == 0:  # Check if the tensor is empty
    #         return None

    #     if self.batch_integ == 'inter':
    #         sets = [set(tensor.tolist()) for tensor in prune_channels]
    #         if len(sets) == 0:
    #             sets = [set()]
    #         intersected_set = set.intersection(*sets)
    #         prune_channels = torch.tensor(list(intersected_set), dtype=torch.long)
    #     elif self.batch_integ == 'union':
    #         sets = [set(tensor.tolist()) for tensor in prune_channels]
    #         if len(sets) == 0:
    #             sets = [set()]
    #         intersected_set = set.union(*sets)
    #         prune_channels = torch.tensor(list(intersected_set), dtype=torch.long)
    #     elif self.batch_integ == 'full':
    #         prune_channels = torch.tensor(prune_channels[0].clone().detach(), dtype=torch.long)
    #     else:
    #         raise ValueError('Not valid batch integration method')
    #     if prune_channels.numel() == 0:
    #         return None
    #     if prune_channels.numel() >= cur_total_channels:
    #         prune_channels_list = prune_channels.tolist()
    #         prune_channels_list.remove(random.choice(prune_channels_list))
    #         # Convert back to tensor
    #         prune_channels = torch.tensor(prune_channels_list, dtype=torch.long)
    #         warnings.warn("Attempting to prune all channels. Keeping one channel for calculation.")
    #     return prune_channels
    
    # def apply_pruning(self, h, key, layer_info, prune_dim, is_prune_out_dim):
    #     # print('apply_pruning', prune_dim, h.dim(), h.shape)
    #     if prune_dim >= h.dim():
    #         raise ValueError('Not valid pruning dimension')
    #         # prune_dim = h.dim() - 1
    #     # No pruning

    #     if 'pqstruct' in self.prune_name:
    #         prune_channels = self.pq_struct(h, key, layer_info, prune_dim)
    #         if self.prune_hyper == 9999:
    #             return [torch.empty(0)]
    #         return prune_channels
    #     elif 'magstruct' in self.prune_name:
    #         if self.prune_hyper == 9999:
    #             return [torch.empty(0)]
    #         prune_channels = self.mag_struct(h, key, layer_info, prune_dim, is_prune_out_dim)
    #         return prune_channels
    #     # elif 'magunstruct' in self.prune_name:
    #     #     pruned_indices = self.mag_unstruct(h, key)
    #     #     return pruned_indices
    #     else:
    #         raise ValueError('Not valid pruning method')
        

    # def pq_struct(self, h, key, layer_info, prune_dim):
    #     info = {}
        
    #     dims_to_aggregate = tuple(i for i in range(h.dim()) if i != prune_dim and i != self.exclude_dim_to_aggregate)
    #     if self.prune_metric == 'IF1N':
    #         norm_across_other_dims = torch.linalg.vector_norm(h, ord=1, dim=dims_to_aggregate)
    #     elif self.prune_metric == 'IF2N':
    #         norm_across_other_dims = torch.linalg.vector_norm(h, ord=2, dim=dims_to_aggregate)    

    #     if 'w*pqstruct' in self.prune_name:
    #         if self.weight_norm_across_channel_dims is None:
    #             self.cal_weight_norm_across_channel_dims(layer_info['weight'])
    #             start_time = time.time()
    #             info[f"{key}_weight_norm_across_channel_dims"] = self.weight_norm_across_channel_dims.tolist()
    #             self.logger_info_time_used += time.time() - start_time
    #         norm_across_other_dims = norm_across_other_dims * self.weight_norm_across_channel_dims
    #     # elif 'probe' in cfg['prune_name'] and cfg['prune_metric'] == 'WIFN':
    #     #     batch_size = h.shape[0]
    #     #     # self.scaler_in *= self.nsamples / (self.nsamples + batch_size)
    #     #     # self.scaler_in += torch.linalg.vector_norm(h, ord=2, dim=dims_to_aggregate) ** 2 / (self.nsamples + batch_size)
    #     #     # norm_across_other_dims = torch.abs(layer_info['weight']) * torch.sqrt(self.scaler_in.reshape((1,-1))).mean(axis=1)

    #     #     self.scaler_in += torch.linalg.vector_norm(h, ord=2, dim=dims_to_aggregate)
    #     #     norm_across_other_dims = (torch.abs(layer_info['weight']) * self.scaler_in.reshape((1,-1))).mean(axis=1)
    #     #     # info[f"{key}_weight_norm_across_channel_dims"] = list(layer_info['weight_norm_across_channel_dims'])
        
    #     # print('norm_across_other_dims', norm_across_other_dims.shape, norm_across_other_dims.dim())
    #     # non_zero_mask = norm_across_other_dims != 0
    #     norm_across_other_dims = norm_across_other_dims + (norm_across_other_dims == 0) * 1e-9
    #     if norm_across_other_dims.dim() == 0 or norm_across_other_dims.dim() == 1:
    #         norm_across_other_dims.unsqueeze_(0)
    #     # print('norm_across_other_dims', norm_across_other_dims, norm_across_other_dims.shape, norm_across_other_dims.dim())
    #     # Calculate norms only for non-zero channels
    #     # non_zero_norms = norm_across_other_dims[non_zero_mask]
    #     norm_p = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_p, dim=1)
    #     norm_q = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_q, dim=1) + 1e-10
    #     # print(self.exclude_dim_to_aggregate, dims_to_aggregate, 1, norm_p, norm_q)
    #     dimension = norm_across_other_dims[0].shape[0]
    #     pq_indices = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * (norm_p / norm_q))

    #     # add additional dimension if dimension is 0
    #     if pq_indices.dim() == 0 or pq_indices.dim() == 1:
    #         pq_indices.unsqueeze_(0)

    #     if torch.isnan(pq_indices).any():
    #         raise ValueError('pq_indices contains nan values')

    #     lower_bound = dimension * (1 + self.eta) ** (-self.pq_q / (self.pq_q - self.pq_p)) * (1 - pq_indices) ** (self.pq_q * self.pq_p / (self.pq_q - self.pq_p))
    #     beta_tensor = torch.full_like(lower_bound, self.pq_beta)
    #     prune_channels_count = torch.floor(dimension * torch.min(self.pq_gamma * (1 - lower_bound / dimension), beta_tensor))
    #     # print('lower_bound', lower_bound.shape, lower_bound)
    #     # print('pq_indices', pq_indices.shape, pq_indices)
    #     sorted_norm, sorted_channels = torch.sort(norm_across_other_dims, dim=1)

    #     eta_zero_lower_bound = dimension * (1 + 0) ** (-self.pq_q / (self.pq_q - self.pq_p)) * (1 - pq_indices) ** (self.pq_q * self.pq_p / (self.pq_q - self.pq_p))
    #     eta_zero_lower_bound = torch.floor(dimension * self.pq_gamma * (1 - eta_zero_lower_bound / dimension))
    #     # print('sorted_norm', sorted_norm.shape, sorted_norm)
    #     if self.prune_hyper == 9999 and cfg['batch_size'] == 1:
    #         # print('here', self.prune_hyper)
    #         start_time = time.time()

    #         def parallel_cal_varying_length_norm(sorted_norm, norm):
    #             if norm == 1:
    #                 # Take the absolute value of each element
    #                 processed_channels = sorted_norm.abs()
    #                 varying_vector_norm = processed_channels.cumsum(dim=1)
    #             elif norm == 2:
    #                 # Take the square of each element
    #                 processed_channels = sorted_norm.pow(2)
    #                 # print('processed_channels', processed_channels.shape, processed_channels[0])
    #                 varying_vector_norm = processed_channels.cumsum(dim=1).sqrt()
    #                 # print('varying_vector_norm', varying_vector_norm.shape, varying_vector_norm[0])
    #             else:
    #                 # Handle other cases or throw an error
    #                 raise ValueError('Not valid norm')
    #             return varying_vector_norm
            
    #         def parallel_cal_varying_length_info(sorted_norm, reversed=False):
    #             if reversed:
    #                 sorted_norm = torch.flip(sorted_norm, [1])
    #             nominator_varying_vector_norm = parallel_cal_varying_length_norm(sorted_norm, self.pq_p)
    #             denominator_varying_vector_norm = parallel_cal_varying_length_norm(sorted_norm, self.pq_q)

    #             nominator_varying_vector_norm = nominator_varying_vector_norm.to(cfg['device'])
    #             denominator_varying_vector_norm = denominator_varying_vector_norm.to(cfg['device'])
    #             # print('nominator_varying_vector_norm', nominator_varying_vector_norm.shape, nominator_varying_vector_norm[0])
    #             # print('denominator_varying_vector_norm', denominator_varying_vector_norm.shape, denominator_varying_vector_norm[0])

    #             num_rows, num_cols = nominator_varying_vector_norm.shape

    #             # if reversed:
    #             #     # Create a tensor where each row starts from 1 and decreases to the length of the row
    #             #     dimension = torch.arange(num_cols, 0, -1).unsqueeze(0)
    #             # else:
    #                 # Create a tensor where each row starts from 1 and increases to the length of the row
    #             dimension = torch.arange(1, num_cols + 1).unsqueeze(0)
    #             dimension = dimension.expand(num_rows, -1).to(cfg['device'])
    #             return nominator_varying_vector_norm, denominator_varying_vector_norm, dimension
            
    #         nominator_varying_vector_norm, denominator_varying_vector_norm, dimension = parallel_cal_varying_length_info(sorted_norm)
    #         # print('dimension', dimension.shape, dimension)
    #         pq_indices_varying_length = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * (nominator_varying_vector_norm / denominator_varying_vector_norm))
            
    #         reversed_nominator_varying_vector_norm, reversed_denominator_varying_vector_norm, dimension = parallel_cal_varying_length_info(sorted_norm, reversed=True)

    #         reversed_pq_indices_varying_length = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * (reversed_nominator_varying_vector_norm / reversed_denominator_varying_vector_norm))
            
    #         # print('nominator_varying_vector_norm', nominator_varying_vector_norm.shape, nominator_varying_vector_norm[0])
    #         # print('denominator_varying_vector_norm', denominator_varying_vector_norm.shape, denominator_varying_vector_norm[0])
            
    #         # exclude length 1 vector
    #         pq_indices_varying_length_cut = pq_indices_varying_length[:, 1:pq_indices_varying_length.shape[1]-2]
    #         # flip back, so it is the result counting from right to left
    #         reversed_pq_indices_varying_length_cut = torch.flip(reversed_pq_indices_varying_length, [1])[:, 2:reversed_pq_indices_varying_length.shape[1]-1]
            
    #         nominator_varying_vector_norm_cut = nominator_varying_vector_norm[:, 1:nominator_varying_vector_norm.shape[1]-2]
    #         reversed_nominator_varying_vector_norm_cut = torch.flip(reversed_nominator_varying_vector_norm, [1])[:, 2:reversed_nominator_varying_vector_norm.shape[1]-1]

    #         denominator_varying_vector_norm_cut = denominator_varying_vector_norm[:, 1:denominator_varying_vector_norm.shape[1]-2]
    #         reversed_denominator_varying_vector_norm_cut = torch.flip(reversed_denominator_varying_vector_norm, [1])[:, 2:reversed_denominator_varying_vector_norm.shape[1]-1]

    #         pq_indices_ratio = pq_indices_varying_length_cut / reversed_pq_indices_varying_length_cut
    #         p_norm_ratio = nominator_varying_vector_norm_cut / reversed_nominator_varying_vector_norm_cut
    #         q_norm_ratio = denominator_varying_vector_norm_cut / reversed_denominator_varying_vector_norm_cut
    #         pq_indices_varying_length = pq_indices_varying_length.mean(dim=0).tolist()
            
    #         # print('lower_bound', eta_zero_lower_bound[0][0])
    #         info[f"{key}_pq_lower_bound"] = eta_zero_lower_bound[0][0].item()
    #         info[f"{key}_pq_indices_varying_lengths"] = pq_indices_varying_length
    #         reversed_pq_indices_varying_length = reversed_pq_indices_varying_length.mean(dim=0).tolist()
    #         info[f"{key}_reversed_pq_indices_varying_lengths"] = reversed_pq_indices_varying_length
    #         info[f"{key}_pq_indices_ratio"] = pq_indices_ratio.mean(dim=0).tolist()
    #         info[f"{key}_p_norm_ratio"] = p_norm_ratio.mean(dim=0).tolist()
    #         info[f"{key}_r_trend"] = q_norm_ratio.mean(dim=0).tolist()
    #         info[f"{key}_q_norm_ratio"] = q_norm_ratio.mean(dim=0).tolist()
            
    #         # print('info[f"{key}_pq_indices_varying_lengths"]', info[f"{key}_pq_indices_varying_lengths"])
    #         # print('info[f"{key}_reversed_pq_indices_varying_lengths"]', info[f"{key}_reversed_pq_indices_varying_lengths"])
    #         # print('info[f"{key}_pq_indices_ratio"]', info[f"{key}_pq_indices_ratio"])
    #         # print('info[f"{key}_p_norm_ratio"]', info[f"{key}_p_norm_ratio"])
    #         # print('info[f"{key}_pq_indices_varying_lengths"]', info[f"{key}_pq_indices_varying_lengths"])
    #         self.logger_info_time_used += time.time() - start_time

    #         # pq_indices_varying_lengths = []
    #         # for i in range(sorted_channels.size(0)):
    #         #     sub_varying_length = [1]
    #         #     # Iterate over the lengths
    #         #     for length in range(1, sorted_channels.size(1) + 1):
    #         #         # Slicing the tensor up to the current length
    #         #         # current_norms = sorted_channels.narrow(self.calc_norm_dim, 0, length)

    #         #         current_channels = sorted_channels[i, :length]
    #         #         current_channels_float = current_channels.float()
    #         #         # print('current_channels_float', current_channels_float.shape, current_channels_float)
    #         #         # Calculate norms
    #         #         norm_p_current = torch.linalg.vector_norm(current_channels_float, ord=self.pq_p, dim=0)
    #         #         norm_q_current = torch.linalg.vector_norm(current_channels_float, ord=self.pq_q, dim=0) + 1e-10
    #         #         # print('norm_p_current', norm_p_current.shape, norm_p_current)
    #         #         # Calculate pq_indices for the current length
    #         #         pq_indices_current = (1 - length ** (1/self.pq_q - 1/self.pq_p) * norm_p_current / norm_q_current)
    #         #         # print('pq_indices_current', pq_indices_current.shape, pq_indices_current)
    #         #         # Check for NaN values
    #         #         if torch.isnan(pq_indices_current).any():
    #         #             raise ValueError('pq_indices contains nan values')

    #         #         # Store the pq_indices for the current length
    #         #         sub_varying_length.append(pq_indices_current.item())
    #         #     pq_indices_varying_lengths.append(sub_varying_length)
    #         # # print('pq_indices_varying_lengths', pq_indices_varying_lengths)
    #         # info[f"{key}_pq_indices_varying_lengths"] = np.array(pq_indices_varying_lengths).mean(axis=0).tolist()
    #         # print('2222 info[f"{key}_pq_indices_varying_lengths"]', info[f"{key}_pq_indices_varying_lengths"])
    #         # self.logger_info_time_used += time.time() - start_time

    #     # print('sorted_channels', sorted_channels.shape, sorted_channels, prune_channels_count)
    #     # if 'probe' in cfg['prune_name']:
    #     #     prune_channels = []
    #     # else:
    #     # print('sorted_channels', sorted_channels.shape, sorted_channels, prune_channels_count)
    #     prune_channels = [sorted_channels[i, :int(count.item())] for i, count in enumerate(prune_channels_count)]
    #     logger_norm_across_other_dims = norm_across_other_dims.mean(dim=0).squeeze(0).tolist()
        
            

    #     start_time = time.time()
    #     info[f"{key}_norm_across_other_dims"] = logger_norm_across_other_dims
    #     info[f"{key}_pq_indices"] = pq_indices.mean(dim=0).squeeze(0).tolist()
    #     self.logger_info_time_used += time.time() - start_time
    #     self.update_pruning_info(info)
    #     return prune_channels

    # # def mag_unstruct(self, h):
    # #     flattened_h = h.view(h.size(0), -1)
    # #     norm_along_dim_1 = torch.linalg.vector_norm(flattened_h, ord=self.prune_norm, dim=1)
    # #     _, sorted_indices = torch.sort(norm_along_dim_1, dim=1)
    # #     num_indices_to_prune = int(self.prune_hyper * sorted_indices.size(1))
    # #     # Select the indices to prune (lowest norms)
    # #     pruned_indices = sorted_indices[:, :num_indices_to_prune]
    # #     return pruned_indices

    # def mag_struct(self, h, key,layer_info, prune_dim, is_prune_out_dim):
    #     info = {}
    #     dims_to_aggregate = tuple(i for i in range(h.dim()) if i != prune_dim and i != self.exclude_dim_to_aggregate)
    #     print('mag_struct input', h, flush=True)
    #     if self.prune_metric == 'IF1N':
    #         norm_across_other_dims = torch.linalg.vector_norm(h, ord=1, dim=dims_to_aggregate)
    #     elif self.prune_metric == 'IF2N':
    #         norm_across_other_dims = torch.linalg.vector_norm(h, ord=2, dim=dims_to_aggregate)
    #     elif self.prune_metric == 'WIF1N':
    #         norm_across_other_dims = (torch.linalg.vector_norm(h, ord=1, dim=dims_to_aggregate) * layer_info['weight'].abs()).sum(dim=0)
    #     elif self.prune_metric == 'WIF2N':
    #         norm_across_other_dims = (torch.linalg.vector_norm(h, ord=2, dim=dims_to_aggregate) * layer_info['weight'].abs()).sum(dim=0)
    #         # print('WIF2N', norm_across_other_dims)
    #     elif self.prune_metric == 'WOF1N':
    #         norm_across_other_dims = (torch.linalg.vector_norm(h, ord=1, dim=dims_to_aggregate) * layer_info['weight'].abs()).sum(dim=1)
    #     elif self.prune_metric == 'WOF2N':
    #         norm_across_other_dims = (torch.linalg.vector_norm(h, ord=2, dim=dims_to_aggregate) * layer_info['weight'].abs()).sum(dim=1)

    #     # if norm_across_other_dims.dim() == 0 or norm_across_other_dims.dim() == 1:
    #     #     norm_across_other_dims.unsqueeze_(0)
    #     # print('w*magstruct', self.prune_name)     
    #     # if 'w*magstruct' in self.prune_name:
    #     #     # print('2222w*magstruct', self.prune_name)
    #     #     if self.weight_norm_across_channel_dims is None:
    #     #         self.cal_weight_norm_across_channel_dims(layer_info['weight'])
    #     #         start_time = time.time()
    #     #         info[f"{key}_weight_norm_across_channel_dims"] = self.weight_norm_across_channel_dims.tolist()
    #     #         self.logger_info_time_used += time.time() - start_time
            
    #     #     norm_across_other_dims = norm_across_other_dims * self.weight_norm_across_channel_dims

    #     if norm_across_other_dims.dim() == 0 or norm_across_other_dims.dim() == 1:
    #         norm_across_other_dims.unsqueeze_(0)

    #     prune_channels_count = int(self.prune_hyper * norm_across_other_dims[0].shape[0])
    #     # print('prune_channels_count', prune_channels_count, norm_across_other_dims[0].shape[0], self.prune_hyper)
    #     sorted_values, sorted_channels = torch.sort(norm_across_other_dims, dim=1)

    #     if sorted_channels.dim() == 0 or sorted_channels.dim() == 1:
    #         sorted_channels.unsqueeze_(0)
        
    #     prune_channels = [sorted_channels[i, :prune_channels_count] for i in range(sorted_channels.size(0))]
    #     print('prune_channels_count', prune_channels_count)
    #     print('sorted_values', sorted_values)
    #     print('prune_channels', prune_channels)
    #     self.update_pruning_info(info)
    #     return prune_channels

        

























class WeightPruning(BasePruning):

    def __init__(self, cfg, key):
        BasePruning.__init__(self, cfg)
        self.key = key
        self.target_modules = _get_target_modules(cfg)
        pass
    
    def global_pruning(self, model):
        if self.prune_hyper == 9999 or self.prune_hyper == 0:
            return
        if len(self.prune_dim) > 1 and self.prune_dim[0] != 1:
            raise ValueError('Not valid prune dim')
        import subprocess
        if 'unstruct' in self.prune_name:
            all_weights = []
            for name, module in model.named_modules():
                if _check_target_module_exists(self.target_modules, name):
                    if hasattr(module, 'weight') and module.weight is not None:
                        all_weights.append(module.weight.data.reshape(-1))

                # Run the nvidia-smi command
                nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']).decode()
                print('nvidia_smi_output', nvidia_smi_output)
                # Process the output
                gpu_utilizations = nvidia_smi_output.strip().split('\n')
                gpu_utilizations = [int(utilization) for utilization in gpu_utilizations]

                print("GPU Utilizations:", gpu_utilizations)

            all_weights_vector = torch.cat(all_weights)
            sorted_weights = torch.sort(torch.abs(all_weights_vector)).values
            if 'magunstruct' in self.prune_name:
                # Concatenate all weights and convert to a single vector
                
                # Rank weights by absolute value and find the threshold for pruning
                num_weights_to_prune = int(all_weights_vector.numel() * self.prune_hyper) 
                print('magunstruct', num_weights_to_prune, all_weights_vector.numel())
                # threshold = torch.kthvalue(torch.abs(all_weights_vector), num_weights_to_prune).values
                threshold = sorted_weights[num_weights_to_prune - 1] 
                print('threshold', threshold)
            elif 'pqunstruct' in self.prune_name:
                # raise ValueError('Not valid pruning method')
                norm_p = torch.linalg.vector_norm(all_weights_vector, ord=self.pq_p, dim=0)
                norm_q = torch.linalg.vector_norm(all_weights_vector, ord=self.pq_p, dim=0) + 1e-10
                
                dimension = len(all_weights_vector)
                pq_indices = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * norm_p / norm_q)

                # add additional dimension if dimension is 0
                if pq_indices.dim() == 0:
                    pq_indices = pq_indices.unsqueeze(0)

                if torch.isnan(pq_indices).any():
                    raise ValueError('pq_indices contains nan values')

                lower_bound = dimension * (1 + self.eta) ** (-self.pq_q / (self.pq_q - self.pq_p)) * (1 - pq_indices) ** (self.pq_q * self.pq_p / (self.pq_q - self.pq_p))
                beta_tensor = torch.full_like(lower_bound, self.pq_beta)
                prune_channels_count = torch.floor(dimension * torch.min(self.pq_gamma * (1 - lower_bound / dimension), beta_tensor))
                # threshold = torch.kthvalue(torch.abs(all_weights_vector), prune_channels_count).values
                threshold = sorted_weights[num_weights_to_prune - 1] 


            # Apply pruning directly without a separate mask
            index = 0
            for name, module in model.named_modules():
                if index >= all_weights_vector.numel():
                    break   
                if _check_target_module_exists(self.target_modules, name):
                    if hasattr(module, 'weight') and module.weight is not None:
                        # numel = module.weight.data.numel()
                        absolute_weights = module.weight.data.abs()

                        # Determine the mask for weights to prune (weights below the threshold)
                        prune_mask = absolute_weights < threshold

                        # Apply pruning to the original weights
                        # This sets weights below the threshold to zero
                        module.weight.data[prune_mask] = 0
                        # module.weight.data.reshape(-1).abs().clamp_(min=threshold)
                        index += torch.sum(prune_mask)
    
                        info = {f"{name}_pruned_ratio": torch.sum(prune_mask).item() / (module.weight.data.numel() + 1e-10)}
                        module.pruning_module.update_pruning_info(info)

            del all_weights_vector
            del all_weights

        elif 'struct' in self.prune_name:
            all_weights = []
            # TODO, prune_dim can only has 1 dim now
            channel_norms = []
            info = {}
            for name, module in model.named_modules():
                is_exist = _check_target_module_exists(self.target_modules, name)
                if _check_target_module_exists(self.target_modules, name):
                    # print('is_exist', is_exist, name, hasattr(module, 'weight'), module.weight is not None, module.weight.data.dim())
                    if hasattr(module, 'weight') and module.weight is not None:
                        dims_to_aggregate = tuple(i for i in range(module.weight.data.dim()) if i != self.prune_dim[0])
                        # print('dims_to_aggregate', dims_to_aggregate)
                        if self.prune_metric == 'WF1N':
                            norm_across_other_dims = torch.linalg.vector_norm(module.weight.data, ord=1, dim=dims_to_aggregate)
                        elif self.prune_metric == 'WF2N':
                            norm_across_other_dims = torch.linalg.vector_norm(module.weight.data, ord=2, dim=dims_to_aggregate)
                        for i, norm in enumerate(norm_across_other_dims):
                            channel_norms.append((norm.item(), name, i, module.weight.shape))  # Store norm, layer name, channel index, and shape
                        info[f"{name}_weight_norm_across_channel_dims"] = norm_across_other_dims.tolist()
            # print('channel_norms', channel_norms)
            if 'magstruct' in self.prune_name:
                channel_norms.sort(key=lambda x: x[0])
                num_channels_to_prune = int(len(channel_norms) * self.prune_hyper) 
                pruning_threshold = channel_norms[num_channels_to_prune - 1][0] if num_channels_to_prune > 0 else float('inf')
            elif 'pqstruct' in self.prune_name:
                norm_p = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_p, dim=0)
                norm_q = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_q, dim=0) + 1e-10
                
                dimension = len(channel_norms)
                pq_indices = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * norm_p / norm_q)

                # add additional dimension if dimension is 0
                if pq_indices.dim() == 0:
                    pq_indices = pq_indices.unsqueeze(0)

                if torch.isnan(pq_indices).any():
                    raise ValueError('pq_indices contains nan values')

                lower_bound = dimension * (1 + self.eta) ** (-self.pq_q / (self.pq_q - self.pq_p)) * (1 - pq_indices) ** (self.pq_q * self.pq_p / (self.pq_q - self.pq_p))
                beta_tensor = torch.full_like(lower_bound, self.pq_beta)
                prune_channels_count = torch.floor(dimension * torch.min(self.pq_gamma * (1 - lower_bound / dimension), beta_tensor))
                pruning_threshold = channel_norms[prune_channels_count - 1][0] if num_channels_to_prune > 0 else float('inf')

            pruning_info = collections.defaultdict(list)
            for norm_value, layer_name, channel_index, shape in channel_norms:
                # print('norm_value', norm_value, 'pruning_threshold', pruning_threshold, 'layer_name', layer_name, 'channel_index', channel_index, 'shape', shape)
                if norm_value > pruning_threshold:
                    break  # Only consider channels below the threshold
                pruning_info[layer_name].append(channel_index)

            # print('pruning_info', pruning_info)
            # Step 2: Apply Pruning
            for name, module in model.named_modules():
                if _check_target_module_exists(self.target_modules, name):
                    # print('is_exist', is_exist, name, hasattr(module, 'weight'), module.weight is not None, module.weight.data.dim())
                    if hasattr(module, 'weight') and module.weight is not None:
                        if name not in pruning_info:
                            module.pruning_module.pruning_info[f"{layer_name}_pruned_ratio"] = 0
                            continue

                        with torch.no_grad():
                            # Prune the specific channel
                            prune_dim = self.prune_dim[0]
                            channel_index_list = pruning_info[name]
                            # info = {f"{name}_pruned_channels": channel_index_list}
                            # module.pruning_module.pruning_info[f"{name}_pruned_ratio"] = len(channel_index_list) / (module.weight.data.shape[prune_dim] + 1e-10)
                            module.pruning_module.update_pruning_info(info)
                            module.weight.data = self.prune_w(module.weight.data, prune_dim, channel_index_list)
                            # Store the pruned channel index if needed
                            module.input_prune_channels = channel_index_list
                            
            
        torch.cuda.empty_cache()

    def local_pruning(self, w, layer_type, layer_info, key, h = None):
        if self.prune_hyper == 9999 or self.prune_hyper == 0:
            return w, None, None
        # if h and 'wanda' not in self.prune_name:
        #     return h, None, None
        if self.batch_integ != 'full':
            raise ValueError('Not valid batch integration method')
        if layer_type == 'linear' and self.prune_dim != [1]:
            raise ValueError('Not valid pruning dimension')
        if max(self.prune_dim) >= w.dim():
            raise ValueError('Not valid pruning dimension')
        
        prune_channels_multi_dims = [None] * w.dim()
        saving_flops_multi_dims = [0] * w.dim()
        w_shape = w.shape
        w_type = w.dtype
        final_prune_dim = 1

        if 'unstruct' in self.prune_name:
            if self.batch_integ in ['inter', 'union']:
                raise ValueError('Not valid batch integration method')
            if 'magunstruct' in self.prune_name:
                flattened_w = w.view(-1)
                # Assuming self.prune_norm is 1 or 2, which are common
                # For L1 norm (or L2 norm), the norm of each element in a 1D vector is its absolute value
                norm = flattened_w.abs()

                # Sort the absolute values
                _, sorted_indices = torch.sort(norm, descending=True)

                # Determine the number of indices to prune
                num_indices_to_prune = int(self.prune_hyper * sorted_indices.size(0))
                # Select the indices to prune (lowest norms)
                prune_indices = sorted_indices[:num_indices_to_prune]
                # Create a mask with the same shape as h, initially set to True
                mask_flattened = torch.ones_like(flattened_w, dtype=torch.bool)
                if prune_indices is not None:
                    # Mark the indices to be pruned as False
                    mask_flattened[prune_indices] = False
                pruned_flattened_w = flattened_w * mask_flattened
                pruned_w = pruned_flattened_w.view_as(w).to(w.device)
                return pruned_w, None, None
            elif 'pqunstruct' in self.prune_name:
                pass

            # elif 'wandaunstruct' in self.prune_name:
            #     if layer_type == 'linear':
            #         dim = (0, 1)
            #         h_norm = torch.linalg.vector_norm(h, ord=2, dim=dim)
            #         h_norm = h_norm.view(1, -1)
            #     elif layer_type == 'conv2d':
            #         raise ValueError('Not valid layer type conv2D')
            #         # dim = (0, 2, 3)
            #         # h_norm = torch.linalg.vector_norm(h, ord=2, dim=dim)
            #         # h_norm = h_norm.view(1, -1, 1, 1)
            #     metric = layer_info['weight'].abs() * h_norm
            #     _, sorted_idx = torch.sort(metric, dim=1) 
            #     pruned_idx = sorted_idx[:,:int(layer_info['weight'].size(1) * self.prune_hyper)] 
            #     layer_info['weight'].scatter_(dim=1, index=pruned_idx, src=0)
            #     # pruned_dims
            #     # prune_channels_multi_dims
            #     return h, 'unstruct', None
        elif 'struct' in self.prune_name:
            if self.prune_dim_select_mode == 'max':
                for prune_dim in self.prune_dim:
                    prune_channels = self.apply_pruning(w, key, prune_dim)
                    prune_channels_multi_dims[prune_dim] = prune_channels
                if len(self.prune_dim) == 1:
                    final_prune_dim = self.prune_dim[0]
                else:
                    raise ValueError('Not valid prune dim')

                pruned_w = self.prune_w(w, final_prune_dim, prune_channels_multi_dims[final_prune_dim])

                pruned_dims = [dim if dim == final_prune_dim else None for dim in range(w.dim())]
                for dim in range(len(prune_channels_multi_dims)):
                    if dim != final_prune_dim:
                        prune_channels_multi_dims[dim] = None
                        saving_flops_multi_dims[dim] = 0
                
                # TODO: hardcode
                if prune_channels_multi_dims[pruned_dims[final_prune_dim]] == None:
                    num_pruned_channels = 0
                else:
                    num_pruned_channels = prune_channels_multi_dims[pruned_dims[final_prune_dim]].size(-1)

                start_time = time.time()
                cur_weight_info = {
                    f"{key}_pruned_dims": pruned_dims[final_prune_dim],
                    # f"{key}_pruned_channels": list(prune_channels_multi_dims[pruned_dims[final_prune_dim]]),
                    f"{key}_total_channels": w_shape[pruned_dims[final_prune_dim]],
                    f"{key}_pruned_ratio": num_pruned_channels / (w_shape[pruned_dims[final_prune_dim]] + 1e-10),
                }
                self.logger_info_time_used += time.time() - start_time
                self.update_pruning_info(cur_weight_info)

                return pruned_w, pruned_dims, prune_channels_multi_dims
        else:
            raise ValueError('Not valid pruning method')

        torch.cuda.empty_cache()

    def prune_w(self, w, prune_dim, prune_channels, mask=None):
        # Create a boolean mask for all indices
        mask = torch.ones(w.size(prune_dim), dtype=torch.bool)
        # Mark the indices to be pruned as False
        if prune_channels is not None:
            mask[prune_channels] = False
        # Use the mask to index the tensor
        pruned_w = w.index_select(dim=prune_dim, index=mask.nonzero().squeeze().to(w.device))
        return pruned_w

        
    def apply_pruning(self, w, key, prune_dim=None):
        if prune_dim >= w.dim():
            raise ValueError('Not valid pruning dimension')
        
        if 'pqstruct' in self.prune_name:
            prune_channels = self.pq_struct(w, key, prune_dim)
            return prune_channels
        elif 'magstruct' in self.prune_name:
            prune_channels = self.mag_struct(w, key, prune_dim)
            return prune_channels
        # elif 'magunstruct' in self.prune_name:
        #     pruned_indices = self.mag_unstruct(h)
        #     return pruned_indices
        else:
            raise ValueError('Not valid pruning method')
        

    def pq_struct(self, w, key, prune_dim):

        calc_dim = 0
        
        dims_to_aggregate = tuple(i for i in range(w.dim()) if i != prune_dim)
        if self.prune_metric == 'WF1N':
            norm_across_other_dims = torch.linalg.vector_norm(w, ord=1, dim=dims_to_aggregate)
        elif self.prune_metric == 'WF2N':
            norm_across_other_dims = torch.linalg.vector_norm(w, ord=2, dim=dims_to_aggregate)   
        norm_across_other_dims = norm_across_other_dims + (norm_across_other_dims == 0) * 1e-9
        norm_p = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_p, dim=calc_dim)
        norm_q = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_q, dim=calc_dim) + 1e-10
        
        dimension = w.shape[prune_dim]
        pq_indices = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * norm_p / norm_q)

        # add additional dimension if dimension is 0
        if pq_indices.dim() == 0:
            pq_indices = pq_indices.unsqueeze(0)

        if torch.isnan(pq_indices).any():
            raise ValueError('pq_indices contains nan values')

        lower_bound = dimension * (1 + self.eta) ** (-self.pq_q / (self.pq_q - self.pq_p)) * (1 - pq_indices) ** (self.pq_q * self.pq_p / (self.pq_q - self.pq_p))
        beta_tensor = torch.full_like(lower_bound, self.pq_beta)
        prune_channels_count = torch.floor(dimension * torch.min(self.pq_gamma * (1 - lower_bound / dimension), beta_tensor))

        _, sorted_channels = torch.sort(norm_across_other_dims, dim=calc_dim)
        prune_channels = sorted_channels[:int(prune_channels_count.item())]
        # info = {
        #     f"{key}_norm_across_other_dims": norm_across_other_dims.mean(dim=0).squeeze(0).tolist(),
        #     f"{key}_pq_indices": pq_indices.mean(dim=0).squeeze(0).tolist(),
        # }
        # self.update_pruning_info(info)
        return prune_channels
    
    def mag_struct(self, w, key, prune_dim):
        dims_to_aggregate = tuple(i for i in range(w.dim()) if i != prune_dim)
        if self.prune_metric == 'WF1N':
            norm_across_other_dims = torch.linalg.vector_norm(w, ord=1, dim=dims_to_aggregate)
        elif self.prune_metric == 'WF2N':
            norm_across_other_dims = torch.linalg.vector_norm(w, ord=2, dim=dims_to_aggregate)      
        _, sorted_channels = torch.sort(norm_across_other_dims, dim=0)
        prune_channels_count = int(self.prune_hyper * w.shape[prune_dim])
        prune_channels = sorted_channels[:int(prune_channels_count)]
        return prune_channels
