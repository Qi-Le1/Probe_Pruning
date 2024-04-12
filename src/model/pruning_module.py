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

class BasePruning:
    def __init__(self, cfg):
        self.prune_metric = cfg['prune_metric']
        self.prune_hyper = cfg['prune_hyper'] 
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
        sorted_norm = torch.flip(sorted_norm, [-1])
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
    
    reversed_nominator_varying_vector_norm, reversed_denominator_varying_vector_norm, reversed_dimension = parallel_cal_varying_length_info(sorted_tensor, pq_p, pq_q, reversed=True)
    print('shape', reversed_dimension.shape)
    res = (nominator_varying_vector_norm[-1].expand(reversed_dimension.shape[0]) / reversed_nominator_varying_vector_norm) - 1
    print('res', res)

    res_numpy = res.cpu().numpy() if isinstance(res, torch.Tensor) else res

    # Find positions where 'res' is approximately 0 (accounting for floating-point precision)
    # Note: Using a tolerance for comparison due to the nature of floating-point arithmetic
    zero_positions = np.where(np.isclose(res_numpy, 0, atol=1e-2))[0]

    # Plotting 'res'
    plt.figure(figsize=(10, 6))
    plt.plot(res_numpy, label='res')
    plt.scatter(zero_positions, res_numpy[zero_positions], color='red', label='res = 1', zorder=5)  # Highlight with red dots
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.title('Visualization of res with Points Where res = 1 Highlighted')
    plt.legend()

    # Ensure the output directory exists
    output_dir = 'output/vis/eta'
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    # key = 'your_key_here'  # Make sure you define 'key' appropriately
    output_path = os.path.join(output_dir, f'{key}_res_plot.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")

    # * reversed_dimension ** (1/pq_p - 1/pq_q))
    # lower_bound = lower_bound.cpu().numpy()
    # x = list(range(len(lower_bound.tolist())))
    # dx = np.diff(x)
    # dy = np.diff(lower_bound)

    # # Compute slope
    # slopes = dy / dx
    
    # if 'low' in cfg['prune_method']:
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

    if 'low' in cfg['prune_method']:
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
    elif 'high' in cfg['prune_method']:
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




def cal_mean_intersection_ratio(first_indices, second_indices):
    if first_indices.dim() == 1:
        first_indices = first_indices.unsqueeze(0)
    bsz = first_indices.shape[0]
    intersection_ratios = []
    for i in range(bsz):
        set_first = set(first_indices[i].cpu().numpy())
        set_second = set(second_indices.cpu().numpy().flatten())
        intersection = set_first.intersection(set_second)
        intersection_ratio = len(intersection) / len(set_first)
        intersection_ratios.append(intersection_ratio)
    return sum(intersection_ratios) / len(intersection_ratios)

def cal_intersection_ratio(output, probe, weight, pruning_module, multiple):

    bsz = output.shape[0]
    in_size = weight.shape[1]
    out_size = weight.shape[0]
    # for each sample
    optimal_norm_squared = torch.clamp(torch.linalg.vector_norm(output, ord=2, dim=1) ** 2, min=cfg['data_type_min_positive'], max=cfg['data_type_max'])
    optimal_weight = weight.unsqueeze(0)
    optimal_dim_metric = torch.sqrt(((optimal_norm_squared.unsqueeze_(1).reshape((-1, 1, in_size))) * torch.pow(optimal_weight, 2)).sum(dim=1).clamp(min=cfg['data_type_min_positive'], max=cfg['data_type_max']))
    # print('optimal_dim_metric', optimal_dim_metric.shape)
    optimal_out_dim_indices, optimal_prune_out_dim_indices = pruning_module.sort_mlp_metric_parallel(optimal_dim_metric, multiple)
    # print('optimal_probe_out_dim_indices', optimal_out_dim_indices.shape, optimal_prune_out_dim_indices.shape, optimal_prune_out_dim_indices)
    fullinf_metric = cal_prune_metric(output, weight, cfg['prune_metric'])
    fullinf_probe_out_dim_indices, fullinf_prune_out_dim_indices = pruning_module.sort_mlp_metric(fullinf_metric, multiple)
    
    fullinf_vs_optimal_select_mean_intersection_ratio = cal_mean_intersection_ratio(optimal_out_dim_indices, fullinf_probe_out_dim_indices)
    fullinf_vs_optimal_prune_mean_intersection_ratio = cal_mean_intersection_ratio(optimal_prune_out_dim_indices, fullinf_prune_out_dim_indices)

    probe_metric = cal_prune_metric(probe, weight, cfg['prune_metric'])
    probe_out_dim_indices, prune_out_dim_indices = pruning_module.sort_mlp_metric(probe_metric, multiple)
    probe_vs_optimal_select_mean_intersection_ratio = cal_mean_intersection_ratio(optimal_out_dim_indices, probe_out_dim_indices)
    probe_vs_optimal_prune_mean_intersection_ratio = cal_mean_intersection_ratio(optimal_prune_out_dim_indices, prune_out_dim_indices)

    probe_vs_fullinf_select_mean_intersection_ratio = cal_mean_intersection_ratio(fullinf_probe_out_dim_indices, probe_out_dim_indices)
    probe_vs_fullinf_prune_mean_intersection_ratio = cal_mean_intersection_ratio(fullinf_prune_out_dim_indices, prune_out_dim_indices)
    # print('cur_bsz_mean_intersection_ratio', cur_bsz_mean_intersection_ratio, 'probe_mean_intersection_ratio', probe_mean_intersection_ratio)
    return fullinf_vs_optimal_select_mean_intersection_ratio, probe_vs_optimal_select_mean_intersection_ratio, probe_vs_fullinf_select_mean_intersection_ratio, \
        fullinf_vs_optimal_prune_mean_intersection_ratio, probe_vs_optimal_prune_mean_intersection_ratio, probe_vs_fullinf_prune_mean_intersection_ratio


# def cal_prune_metric(probe_out, weight, metric_type, global_metric_score_distribution=None, global_input_distribution=None, selected_indices=None):
#     # if probe_out.size(0) != 1 and probe_out.size(0) != cfg['probe_num']:
#     #     raise ValueError('probe_out size in calculating metric should be 1 or probe_num')
#     print('probe_out shape', probe_out.shape)
#     probe_num = probe_out.size(0)
#     if 'wandasp' in metric_type:
#         # if probe_out.dim() == 2:
#         #     probe_out.unsqueeze_(0)
#         size = probe_out.shape[0]
#         # sum_squared_norms = torch.sum(torch.linalg.vector_norm(probe_out, ord=2, dim=1) ** 2 * weight_factor, dim=0)
#         norm_squared = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=(0, 1)) ** 2, min=None, max=65504) / size
#         if global_metric_score_distribution is not None:
#             norm_squared = cfg['ema_momentum'] * global_metric_score_distribution.to(probe_out.device) + (1 - cfg['ema_momentum']) * norm_squared
#         probe_out_dim_metric = (torch.sqrt(norm_squared.unsqueeze_(0).reshape((1,-1))) * torch.abs(weight)).sum(dim=0)
#     elif 'flap' in metric_type:
#         pass
#     elif 'probe' in metric_type:
#         # self.scaler_inp = self.scaler_inp.to(cur_device)
#         # self.nsamples = self.nsamples.to(cur_device)
#         # update_indices = update_indices.to(cur_device)
#         # self.scaler_inp[update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + batch_size)
#         # norm_squared = torch.clamp(torch.linalg.vector_norm(inp, ord=2, dim=1) ** 2, min=None, max=65504)
#         # # the probe for batch size, modify the denominator
#         # # if is_probe:
#         # #     denominator = (self.nsamples[update_indices].unsqueeze(0) + inp.shape[0])
#         # # else:
#         # denominator = (self.nsamples[update_indices].unsqueeze(0) + batch_size)
#         # self.scaler_inp[update_indices] += torch.sum(norm_squared / denominator, dim=0)
#         combined_probe_out = None
#         # if probe_out.dim() == 2:
#         #     probe_out.unsqueeze_(0)
#         # print('probe_out', probe_out.shape, probe_out)
#         # probe_size = probe_out.shape[0]
#         # if size != 1:
#         #     raise ValueError('probe_out size should be 1')
#         # if global_input_distribution is not None:
#         #     # print('norm_squared', norm_squared.shape, norm_squared)
#         #         # print('global_metric_score_distribution', global_metric_score_distribution.shape, global_metric_score_distribution)
#         #     ema = cfg['ema_momentum'] * global_input_distribution.to(probe_out.device) + (1 - cfg['ema_momentum']) * probe_out
#         #         # print('norm_squared new', norm_squared.shape, norm_squared)
#         #     norm_squared = torch.clamp(torch.linalg.vector_norm(ema, ord=2, dim=(0, 1)) ** 2, min=None, max=65504) / size
#         #     probe_out_dim_metric = torch.sqrt(((norm_squared.unsqueeze_(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(min=None, max=65504))
#         # else:
#         # if 'savemetricseq' in cfg['prune_method']:
            
#             # norm_probe_out = torch.linalg.vector_norm(probe_out, ord=2, dim=0)
#         if global_metric_score_distribution is not None:
#             norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, min=cfg['data_type_min_positive'], max=65504)
#             global_metric_score_distribution = torch.clamp(global_metric_score_distribution.to(probe_out.device), min=cfg['data_type_min_positive'], max=65504)
#             # norm_probe_out = cfg['fix_seq_merge_ratio'] * global_metric_score_distribution.to(probe_out.device) + (1 - cfg['fix_seq_merge_ratio']) * norm_probe_out
#             # if 'probefixratio' in cfg['prune_method']:
#             #     # denominator = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 + global_metric_score_distribution.to(probe_out.device), min=None, max=65504)
#             #     # global_ratio =  global_metric_score_distribution.to(probe_out.device) / (denominator + 1e-10)
#             #     # probe_ratio = torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 /(denominator + 1e-10)
#             #     combined_probe_out = cfg['probefixratio'] * global_metric_score_distribution.to(probe_out.device) + (1-cfg['probefixratio']) * (norm_probe_out_square)
#             #     norm_probe_out = torch.sqrt(combined_probe_out)
#             #     # norm_probe_out = cfg['ema_momentum'] * torch.sqrt(global_metric_score_distribution.to(probe_out.device)) + (1 - cfg['ema_momentum']) * torch.linalg.vector_norm(probe_out, ord=2, dim=0) 
#             # # if 'savemetricseqv1ratio' in cfg['prune_method']:
#             # #     denominator = torch.linalg.vector_norm(probe_out, ord=2, dim=0) + torch.sqrt(global_metric_score_distribution.to(probe_out.device))
#             # #     global_ratio =  torch.sqrt(global_metric_score_distribution.to(probe_out.device)) / (denominator + 1e-10)
#             # #     probe_ratio = torch.linalg.vector_norm(probe_out, ord=2, dim=0) / (denominator + 1e-10)
#             # #     norm_probe_out = global_ratio * global_metric_score_distribution.to(probe_out.device) + probe_ratio * torch.linalg.vector_norm(probe_out, ord=2, dim=0)
#             # elif 'probedynaratio' in cfg['prune_method']:
#             denominator = norm_probe_out_square + global_metric_score_distribution.to(probe_out.device)
#             probe_ratio = norm_probe_out_square / denominator
#             has_nan = torch.isnan(probe_ratio).any()
#             has_inf = torch.isinf(probe_ratio).any()
#             if has_nan:
#                 print("Does 'probe_ratio' contain NaN values? Yes")
#             if has_inf:
#                 print("Does 'probe_ratio' contain infinite values? Yes")
#             # global_metric_score_distribution.to(probe_out.device) / (denominator + 1e-10)
#             global_ratio = 1 - probe_ratio
#             has_nan = torch.isnan(global_ratio).any()
#             has_inf = torch.isinf(global_ratio).any()

#             # Printing checks for NaN and Inf
#             if has_nan:
#                 print("Does 'global_ratio' contain NaN values? Yes")
#             if has_inf:
#                 print("Does 'global_ratio' contain infinite values? Yes")
#             combined_probe_out = global_ratio * global_metric_score_distribution.to(probe_out.device) + probe_ratio * norm_probe_out_square
#             norm_probe_out = torch.sqrt(combined_probe_out)

#             temp_sort_value, _ = torch.sort(norm_probe_out_square, dim=-1)
#             torch.set_printoptions(threshold=5000)
#             print('norm_probe_out_square', norm_probe_out_square.shape, temp_sort_value)
#             temp_sort_value, _ = torch.sort(global_metric_score_distribution.to(probe_out.device), dim=-1)
#             print('global_metric_score_distribution', global_metric_score_distribution.shape, temp_sort_value)

#             print('globaratio', global_ratio, torch.mean(global_ratio))
#             # elif 'probeadd' in cfg['prune_method']:
#             #     norm_probe_out = torch.sqrt(torch.clamp(norm_probe_out_square + global_metric_score_distribution.to(probe_out.device), min=None, max=65504))
#             #     # norm_probe_out = cfg['ema_momentum'] * torch.sqrt(global_metric_score_distribution.to(probe_out.device)) + (1 - cfg['ema_momentum']) * torch.linalg.vector_norm(probe_out, ord=2, dim=0) 
#             # elif 'probemax' in cfg['prune_method']:
                
#             #     norm_probe_out = torch.sqrt(torch.max(norm_probe_out_square, global_metric_score_distribution.to(probe_out.device)))
#             #     temp_sort_value, _ = torch.sort(norm_probe_out_square, dim=-1)
#             #     print('norm_probe_out_square', norm_probe_out_square.shape, temp_sort_value)
#             #     temp_sort_value, _ = torch.sort(global_metric_score_distribution.to(probe_out.device), dim=-1)
#             #     print('global_metric_score_distribution', global_metric_score_distribution.shape, temp_sort_value)
            
#             print('norm_probe_out.shaope', norm_probe_out.shape)
#             norm_squared = torch.clamp(torch.linalg.vector_norm(norm_probe_out, ord=2, dim=0) ** 2, min=cfg['data_type_min_positive'], max=65504)
#             # norm_squared = torch.sum(norm_probe_out, dim=0)
#             probe_out_dim_metric = torch.sqrt(((norm_squared.unsqueeze_(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(min=cfg['data_type_min_positive'], max=65504))

#             if 'fillpbmetriccombine' in cfg['prune_method']:
#             # in update, first take l2 then square
#             # take l2 -> sqrt of the sum of square 
#             # take square -> what we want to store
#                 return probe_out_dim_metric, torch.sqrt(combined_probe_out).unsqueeze_(0)
#             else:
#                 return probe_out_dim_metric, None
#         else:
#             norm_probe_out_square = torch.linalg.vector_norm(probe_out, ord=2, dim=(0,1)) ** 2 / probe_num
#             # norm_squared = torch.clamp(torch.linalg.vector_norm(norm_probe_out, ord=2, dim=0) ** 2, min=None, max=65504) / size
#             # norm_squared = torch.sum(norm_probe_out, dim=0)
#             probe_out_dim_metric = torch.sqrt(((norm_probe_out_square.unsqueeze_(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(min=None, max=65504))
#             return probe_out_dim_metric, None
            # combined_probe_out = norm_probe_out
            # else:
            #     norm_squared = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2, min=None, max=65504) / size
            #     probe_out_dim_metric = torch.sqrt(((norm_squared.unsqueeze_(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(min=None, max=65504))
        # else:
        #     norm_probe_out_square = torch.linalg.vector_norm(probe_out, ord=2, dim=(0,1)) ** 2 / probe_num
        #     if global_metric_score_distribution is not None:
        #         # norm_probe_out = cfg['fix_seq_merge_ratio'] * global_metric_score_distribution.to(probe_out.device) + (1 - cfg['fix_seq_merge_ratio']) * norm_probe_out
        #         if 'probefixratio' in cfg['prune_method']:
        #             # denominator = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 + global_metric_score_distribution.to(probe_out.device), min=None, max=65504)
        #             # global_ratio =  global_metric_score_distribution.to(probe_out.device) / (denominator + 1e-10)
        #             # probe_ratio = torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 /(denominator + 1e-10)
        #             combined_probe_out = cfg['probefixratio'] * global_metric_score_distribution.to(probe_out.device) + (1-cfg['probefixratio']) * (norm_probe_out_square)
        #             # norm_probe_out = torch.sqrt(combined_probe_out)
        #             # norm_probe_out = cfg['ema_momentum'] * torch.sqrt(global_metric_score_distribution.to(probe_out.device)) + (1 - cfg['ema_momentum']) * torch.linalg.vector_norm(probe_out, ord=2, dim=0) 
        #         # if 'savemetricseqv1ratio' in cfg['prune_method']:
        #         #     denominator = torch.linalg.vector_norm(probe_out, ord=2, dim=0) + torch.sqrt(global_metric_score_distribution.to(probe_out.device))
        #         #     global_ratio =  torch.sqrt(global_metric_score_distribution.to(probe_out.device)) / (denominator + 1e-10)
        #         #     probe_ratio = torch.linalg.vector_norm(probe_out, ord=2, dim=0) / (denominator + 1e-10)
        #         #     norm_probe_out = global_ratio * global_metric_score_distribution.to(probe_out.device) + probe_ratio * torch.linalg.vector_norm(probe_out, ord=2, dim=0)
        #         elif 'probedynaratio' in cfg['prune_method']:
        #             denominator = torch.clamp(norm_probe_out_square + global_metric_score_distribution.to(probe_out.device), min=None, max=65504)
        #             global_ratio =  global_metric_score_distribution.to(probe_out.device) / (denominator + 1e-10)
        #             probe_ratio = norm_probe_out_square /(denominator + 1e-10)
        #             combined_probe_out = global_ratio * global_metric_score_distribution.to(probe_out.device) + probe_ratio * norm_probe_out_square
        #             # norm_probe_out = torch.sqrt(combined_probe_out)
        #             # norm_probe_out = cfg['ema_momentum'] * torch.sqrt(global_metric_score_distribution.to(probe_out.device)) + (1 - cfg['ema_momentum']) * torch.linalg.vector_norm(probe_out, ord=2, dim=0) 
        #         print('norm_probe_out.shaope', norm_probe_out.shape)
        #         # norm_squared = torch.clamp(torch.linalg.vector_norm(norm_probe_out, ord=2, dim=0) ** 2, min=None, max=65504) / size
        #         # norm_squared = torch.sum(norm_probe_out, dim=0)
        #         probe_out_dim_metric = torch.sqrt(((combined_probe_out.unsqueeze_(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(min=None, max=65504))
        #     # combined_probe_out = norm_probe_out
        #     else:
        #         # norm_squared = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=(0,1)) ** 2, min=None, max=65504) / size
        #         probe_out_dim_metric = torch.sqrt(((norm_probe_out_square.unsqueeze_(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(min=None, max=65504))

        #     if 'fillpbmetriccombine' in cfg['prune_method']:
        #         # in update, first take l2 then square
        #         # take l2 -> still sqrt
        #         # take square -> current value
        #         return probe_out_dim_metric, torch.sqrt(combined_probe_out).unsqueeze_(0).unsqueeze_(0)
        #     else:
        #         return probe_out_dim_metric, None
            
def cal_prune_metric(probe_out, weight, metric_type, global_metric_score_distribution=None, global_input_distribution=None, selected_indices=None):
    probe_num = probe_out.size(0)
    if 'wandasp' in metric_type:
        size = probe_out.shape[0]
        norm_squared = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=(0, 1)) ** 2, max=cfg['data_type_max']) / size
        if global_metric_score_distribution is not None:
            norm_squared = cfg['ema_momentum'] * global_metric_score_distribution.to(probe_out.device) + (1 - cfg['ema_momentum']) * norm_squared
        probe_out_dim_metric = (torch.sqrt(norm_squared.unsqueeze_(0).reshape((1,-1))) * torch.abs(weight)).sum(dim=0)
    elif 'flap' in metric_type:
        pass
    elif 'probe' in metric_type:
        combined_probe_out = None
        if 'probe' in cfg['prune_method']:
            if global_metric_score_distribution is not None:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])
                print('norm_probe_out_square ', norm_probe_out_square.dtype)
                print('global_metric_score_distribution ', global_metric_score_distribution.dtype)
                global_metric_score_distribution = global_metric_score_distribution.to(probe_out.device)
                has_nan = torch.isnan(global_metric_score_distribution).any()
                has_inf = torch.isinf(global_metric_score_distribution).any()
                if has_nan:
                    print("Does 'global_metric_score_distribution' contain NaN values? Yes")
                if has_inf:
                    print("Does 'global_metric_score_distribution' contain infinite values? Yes")

                if 'seq' in cfg['probe_info']:
                    if 'rank' in cfg['probe_info']:
                        global_metric_score_distribution = global_metric_score_distribution[selected_indices, :]
                    elif 'mean' in cfg['probe_info']:
                        # global_metric_score_distribution = mean_process(global_metric_score_distribution, probe_out.size(1), cfg['seq_len']//probe_out.size(1))
                        global_metric_score_distribution = torch.mean(global_metric_score_distribution.view(probe_out.size(1), cfg['seq_len']//probe_out.size(1), global_metric_score_distribution.size(-1)), dim=1)

                if 'probefixratio' in cfg['probe_info']:
                    combined_probe_out = cfg['probefixratio'] * global_metric_score_distribution + (1-cfg['probefixratio']) * norm_probe_out_square
                    # norm_probe_out = torch.sqrt(combined_probe_out)
                # dynaratio
                else:
                    
                            # pass
                        
                    # else:
  
                    denominator = norm_probe_out_square + global_metric_score_distribution
                    print('denominator', denominator.dtype)
                    # global_ratio = norm_probe_out_square / (denominator + 1e-6)
                    # has_nan = torch.isnan(global_ratio).any()
                    # has_inf = torch.isinf(global_ratio).any()
                    # if has_nan:
                    #     print("Does 'global_ratio' contain NaN values? Yes")
                    # if has_inf:
                    #     print("Does 'global_ratio' contain infinite values? Yes")
                    # # global_metric_score_distribution.to(probe_out.device) / (denominator + 1e-10)
                    # probe_ratio = 1 - global_ratio

                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight if there is a nan
                    probe_ratio = norm_probe_out_square / (denominator + 1e-6)
                    has_nan = torch.isnan(probe_ratio).any()
                    has_inf = torch.isinf(probe_ratio).any()
                    if has_nan:
                        print("Does 'probe_ratio' contain NaN values? Yes")
                    if has_inf:
                        print("Does 'probe_ratio' contain infinite values? Yes")
                    # global_metric_score_distribution.to(probe_out.device) / (denominator + 1e-10)
                    global_ratio = 1 - probe_ratio
                    # print('global_ratio', global_ratio, torch.mean(global_ratio))

                    combined_probe_out = global_ratio * global_metric_score_distribution + probe_ratio * norm_probe_out_square

                    # combined_probe_out = torch.max(norm_probe_out_square, global_metric_score_distribution)

                    # Stack the tensors to create a new dimension where a and b are stacked
                    # stacked = torch.stack((norm_probe_out_square, global_metric_score_distribution), dim=2)

                    # # Apply softmax across the new dimension (dim=2) to merge them
                    # combined_probe_out = torch.sum(torch.softmax(stacked, dim=2) * stacked, dim=2)

                    # norm_probe_out = torch.sqrt(combined_probe_out)

                    # temp_sort_value, _ = torch.sort(norm_probe_out_square, dim=-1)
                    # torch.set_printoptions(threshold=5000)
                    # print('norm_probe_out_square', norm_probe_out_square.shape, temp_sort_value)
                    # temp_sort_value, _ = torch.sort(global_metric_score_distribution, dim=-1)
                    # print('global_metric_score_distribution', global_metric_score_distribution.shape, temp_sort_value)
                    # print('globaratio', global_ratio, torch.mean(global_ratio))
                print('combined_probe_out', combined_probe_out.shape)
                combined_probe_out = torch.sum(combined_probe_out, dim=0)
                probe_out_dim_metric = torch.sqrt(((combined_probe_out.unsqueeze_(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(max=cfg['data_type_max']))

                # if 'fillpbmetriccombine' in cfg['prune_method']:
                # # in update, first take l2 then square
                # # take l2 -> sqrt of the sum of square 
                # # take square -> what we want to store
                #     return probe_out_dim_metric, torch.sqrt(combined_probe_out).unsqueeze_(0)
                # else:
                return probe_out_dim_metric, None
            else:
                print('probe_num', probe_num, probe_out.shape)
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=(0,1)) ** 2 / probe_num, max=cfg['data_type_max'])
                probe_out_dim_metric = torch.sqrt(((norm_probe_out_square.unsqueeze_(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(max=cfg['data_type_max']))
                return probe_out_dim_metric, None
        else:
            norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=(0,1)) ** 2 / probe_num,  max=cfg['data_type_max'])
            global_metric_score_distribution = global_metric_score_distribution.to(probe_out.device)
            if global_metric_score_distribution is not None:
                if 'probefixratio' in cfg['probe_info']:
                    combined_probe_out = cfg['probefixratio'] * global_metric_score_distribution + (1-cfg['probefixratio']) * norm_probe_out_square
                else:
                    denominator = norm_probe_out_square + global_metric_score_distribution
                    global_ratio = torch.clamp(norm_probe_out_square / denominator, max=1)
                    probe_ratio = 1 - global_ratio
                    combined_probe_out = global_ratio * global_metric_score_distribution + probe_ratio * norm_probe_out_square
                probe_out_dim_metric = torch.sqrt(((combined_probe_out.unsqueeze_(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(max=cfg['data_type_max']))
            else:
                probe_out_dim_metric = torch.sqrt(((norm_probe_out_square.unsqueeze_(0).reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(max=cfg['data_type_max']))

            # if 'fillpbmetriccombine' in cfg['prune_method']:
            #     # in update, first take l2 then square
            #     # take l2 -> still sqrt
            #     # take square -> current value
            #     return probe_out_dim_metric, torch.sqrt(combined_probe_out).unsqueeze_(0).unsqueeze_(0)
            # else:
            return probe_out_dim_metric, None        



def cal_calib_prune_metric(calib, weight, metric_type):
    if 'wandasp' in metric_type:
        
        probe_out_dim_metric = (torch.sqrt(calib.reshape((1,-1))) * torch.abs(weight)).sum(dim=0)
    elif 'flap' in metric_type:
        pass
    elif 'probe' in metric_type:
        # if 'savemetricseq' in cfg['prune_method']:
        #     calib = torch.sum(calib, dim=0)
        #     probe_out_dim_metric = torch.sqrt(((calib.reshape((1,-1))) * torch.pow(weight, 2)).sum(dim=0).clamp(min=cfg['data_type_min'], max=cfg['data_type_max']))
        # else:
            # print('running_mean', running_mean.shape, running_mean)
        # due to linearity, should be the same as save seq and out dim in calib

        # squared_weight = torch.pow(weight, 2)

        # # Reshape calib using PyTorch's view or reshape method
        # calib_reshaped = calib.view(1, -1)

        # # Use torch.mul for element-wise multiplication, then sum, clamp, and finally compute the square root
        # probe_out_dim_metric = torch.sqrt(torch.mul(calib_reshaped, squared_weight).sum(dim=0).clamp(min=cfg['data_type_min'], max=cfg['data_type_max']))

        # probe_out_dim_metric = torch.sqrt(torch.mul(calib.reshape((1,-1)), torch.pow(weight, 2)).sum(dim=0).clamp(min=cfg['data_type_min'], max=cfg['data_type_max']))

        # Use torch.pow for squaring, torch.mul for multiplication, and torch.sum for summation
        squared_weight = torch.pow(weight, 2)
        calib_reshaped = calib.reshape((1, -1))
        mult_result = torch.mul(calib_reshaped, squared_weight)

        # Use torch.sum instead of .sum(), then apply torch.clamp and finally compute the square root
        probe_out_dim_metric = torch.sqrt(torch.clamp(torch.sum(mult_result, dim=0),  max=cfg['data_type_max']))

    return probe_out_dim_metric



class HiddenRepresentationPruning(BasePruning):

    def __init__(self, cfg, key, device=None, in_dim=None, out_dim=None):
        BasePruning.__init__(self, cfg)
        self.key = key
        self.device = device
        if out_dim:
            self.scaler_in = torch.zeros((in_dim), device=self.device)
        self.nsamples = 0
        if 'pq' in cfg['prune_method']:
            self.pq_p = cfg['pq_p']
            self.pq_q = cfg['pq_q']
            self.eta = cfg['prune_hyper']
            self.pq_beta = cfg['pq_beta']
            self.pq_gamma = cfg['pq_gamma']

    def box_cox_transformation(self, x, lam):
        """
        Apply the Box-Cox Transformation to a tensor x.
        Note: x must be positive (>0).
        """
        if lam == 0:
            return torch.log(x)
        else:
            return (x ** lam - 1) / lam
        
    def sort_mlp_metric(self, probe_out_dim_metric, multiple, pruning_ratio=None):
        self.prune_hyper = pruning_ratio if pruning_ratio is not None else self.prune_hyper
        # probe_out_dim_metric.abs_()
        # probe_out_dim_metric = probe_out_dim_metric.to(torch.float32)
        # mask = torch.ones(probe.shape[-1], dtype=torch.bool, device=probe.device)
        sorted_value, sorted_indices = torch.sort(probe_out_dim_metric, dim=0)
        if cfg['logger_detailed_info'] == True:
            torch.set_printoptions(threshold=1000)
            print(f'{self.key} sorted_value', sorted_value)
            nan_count = torch.isnan(sorted_value).sum().item()
            inf_count = torch.isinf(sorted_value).sum().item()
            if nan_count > 0 or inf_count > 0:
                print(f'{self.key} sorted_value contains {nan_count} NaN and {inf_count} inf values')
            print(f'{self.key} pruning ratio', self.prune_hyper)
        # normalized_sorted_value = sorted_value / sorted_value.sum()
        # print(f'{self.key} normalized_sorted_value', normalized_sorted_value)
        # mean = torch.mean(sorted_value)
        # std = torch.std(sorted_value)

        # # Then, normalize the tensor: (sorted_value - mean) / std
        # standardlized_value = (sorted_value - mean) / std
        # print(f'{self.key} standardlized_value', standardlized_value)
        # num_prune = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma, self.key)[0]            
        if 'pq' in cfg['prune_method']:
            print('mean', torch.mean(sorted_value), 'std', torch.std(sorted_value))
            num_prune = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma, self.key)[0]
        else:
            num_prune = int(self.prune_hyper * probe_out_dim_metric.shape[0])

            # mean = torch.mean(sorted_value)
            # std = torch.std(sorted_value)

            # # Define the bounds for being within 4 standard deviations
            # lower_bound = mean - 3 * std
            # upper_bound = mean + 3 * std

            # # Find points outside of 4 standard deviations
            # low_outliers = sorted_value < lower_bound
            # high_outliers = sorted_value > upper_bound
            # # outliers = ((sorted_value < lower_bound) | (sorted_value > upper_bound))
            # low_outliers_num = torch.sum(low_outliers).item()
            # high_outliers_num = torch.sum(high_outliers).item()
            # print(f'Number of points below 4 standard deviations: {low_outliers_num}')
            # print(f'Number of points above 4 standard deviations: {high_outliers_num}')

            # # 假设sorted_value是你的张量
            # q1 = torch.quantile(sorted_value, 0.25)
            # q3 = torch.quantile(sorted_value, 0.75)
            # iqr = q3 - q1

            # # # 定义异常值的界限为Q1 - 3IQR和Q3 + 3IQR
            # lower_bound = q1 - 3 * iqr
            # upper_bound = q3 + 3 * iqr

            # # # 找到低于或高于界限的点
            # low_outliers = sorted_value < lower_bound
            # high_outliers = sorted_value > upper_bound

            # # 计算低于和高于界限的点的数量
            # low_outliers_num = torch.sum(low_outliers).item()
            # high_outliers_num = torch.sum(high_outliers).item()

            # print(f'Number of points below Q1 - 3IQR: {low_outliers_num}')
            # print(f'Number of points above Q3 + 3IQR: {high_outliers_num}')

            # # Count the number of outliers
            # # num_outliers = torch.sum(outliers).item()

            # # print(f'Number of points outside of 4 standard deviations: {num_outliers}')
            # print(f'{self.key} box_cox_transformation', sorted_value)
            # if high_outliers_num > 0:  # Check to ensure we have outliers to exclude
            #     sorted_value = sorted_value[: -high_outliers_num]
            #     print('shape', sorted_value.shape)
            # else:
                # adjusted_sorted_value = sorted_value
        
        start_time = time.time()
        num_prune = nearest_multiple(num_prune, probe_out_dim_metric.shape[0], multiple)
        # print(f'{self.key} nearest_multiple time', time.time() - start_time)
        return sorted_indices[num_prune:], sorted_indices[:num_prune]
    
    def sort_mlp_metric_parallel(self, probe_out_dim_metric, multiple):
        probe_out_dim_metric.abs_()
        # probe_out_dim_metric = probe_out_dim_metric.to(torch.float32)
        # mask = torch.ones(probe.shape[-1], dtype=torch.bool, device=probe.device)
        sorted_value, sorted_indices = torch.sort(probe_out_dim_metric, dim=-1)
        # print(f'{self.key} sort_mlp_metric_parallel sorted_value', sorted_value)
        # normalized_sorted_value = sorted_value / sorted_value.sum()
        # print(f'{self.key} normalized_sorted_value', normalized_sorted_value)
        # mean = torch.mean(sorted_value)
        # std = torch.std(sorted_value)

        # # Then, normalize the tensor: (sorted_value - mean) / std
        # standardlized_value = (sorted_value - mean) / std
        # print(f'{self.key} standardlized_value', standardlized_value)
        if 'mag' in cfg['prune_method']:
            num_prune = int(self.prune_hyper * probe_out_dim_metric.shape[-1])
        elif 'pq' in cfg['prune_method']:
            num_prune = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma, self.key)[0]

        # let the remaining element be the multiple of multiple to fit tensor cores
        num_prune = num_prune + ((probe_out_dim_metric.shape[0] - num_prune) % multiple)
        return sorted_indices[..., num_prune:], sorted_indices[...,:num_prune]
    
    # def cal_probe_attn_weights_metric(self, attn_weights_metric):
        
    #     attn_weights_metric = attn_weights_metric.to(torch.float32)
    #     sorted_value, sorted_indices = torch.sort(attn_weights_metric, dim=1)
    #     if 'mag' in cfg['prune_method']:
    #         num_prune = int(self.prune_hyper * attn_weights_metric.shape[1])
    #     # Select indices to prune for each head
    #     indices_to_preserve = sorted_indices[:, num_prune:]

    #     return indices_to_preserve

        # attn_weights_metric.abs_()
        # attn_weights_metric = attn_weights_metric.to(torch.float32)
        # # mask = torch.ones(probe.shape[-1], dtype=torch.bool, device=probe.device)
        # sorted_value, sorted_indices = torch.sort(attn_weights_metric, dim=0)
        # if 'mag' in cfg['prune_method']:
        #     num_prune = int(self.prune_hyper * attn_weights_metric.shape[0])
        # elif 'pq' in cfg['prune_method']:
        #     num_prune = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma)[0]
        #     # print('num_prune', num_prune)
        # # print('prune_channels_count', prune_channels_count, norm_across_other_dims[0].shape[0], self.prune_hyper)

        # return sorted_indices[num_prune:]

    def sort_probe_attn_metric(self, probe_out_dim_metric, num_heads, head_dim, prune_way, prune_module, multiple, pruning_ratio=None):
        # mask = torch.ones(probe.shape[-1], dtype=torch.bool, device=probe.device)
        # probe_out_dim_metric.abs_()
        if prune_way == None:
            return None, None, num_heads, head_dim
            return None, None, None, num_heads, head_dim
        
        self.prune_hyper = pruning_ratio if pruning_ratio is not None else self.prune_hyper
        # probe_out_dim_metric = probe_out_dim_metric.to(torch.float32)
        
        if 'eachwhole' in prune_way:
            sorted_probe_out_dim_metric_value, _ = torch.sort(probe_out_dim_metric)
            threshold = sorted_probe_out_dim_metric_value[self.prune_hyper * probe_out_dim_metric.numel()]

            probe_out_dim_metric = probe_out_dim_metric.reshape(num_heads, -1)
            # Sort the probe_out_dim_metric across each head
            sorted_value, sorted_indices = torch.sort(probe_out_dim_metric, dim=1)
            # num_prune_head_dim = int(self.prune_hyper * head_dim)
            # num_prune_head_dim = nearest_multiple(num_prune_head_dim * num_heads, probe_out_dim_metric.numel(), multiple, num_heads) // num_heads
            # indices_to_preserve = sorted_indices[:, num_prune_head_dim:]

            # preserved_dim_values = sorted_value[:, num_prune_head_dim:]
            comparison_result = probe_out_dim_metric <= threshold

            # Sum the boolean values in each row (True is treated as 1, False as 0)
            sum_true_per_row = comparison_result.sum(dim=1)

            # Number of columns in sorted_temp
            num_columns = probe_out_dim_metric.shape[-1]

            # Compare the sum of True values per row to the number of columns
            rows_all_true = sum_true_per_row == num_columns
            rows_not_all_true = sum_true_per_row != num_columns

            print('rows_all_true', rows_all_true.sum().item(), rows_all_true)

            proportion_rows_all_true = rows_all_true.sum().item() / probe_out_dim_metric.shape[0]
            print(f'Proportion of rows entirely True: {proportion_rows_all_true}')

            num_heads = num_heads - rows_all_true.sum().item()


            num_prune_head_dim = int(self.prune_hyper * head_dim)
            num_prune_head_dim = nearest_multiple(num_prune_head_dim * num_heads, num_heads * head_dim, multiple, num_heads) // num_heads
            indices_to_preserve = sorted_indices[:, num_prune_head_dim:]
            sorted_indices[rows_not_all_true, num_prune_head_dim:]

            # for vo
            rows_not_all_true_indices = torch.where(rows_not_all_true)[0]
            head_range = rows_not_all_true_indices * head_dim
            # Create the full indices for pruning using broadcasting
            full_indices_to_preserve_vo = (indices_to_preserve + head_range.unsqueeze(1)).view(-1)

            # for qk:
            full_indices_to_preserve_vo_qk = (torch.arange(head_dim, device=probe_out_dim_metric.device) + rows_not_all_true_indices.unsqueeze(1) * head_dim).view(-1)
            # indices_to_preserve = indices_to_preserve[rows_not_all_true, ...]
            return full_indices_to_preserve_vo, full_indices_to_preserve_vo_qk, None, num_heads, head_dim
        # delete whole head
        elif 'whole' in prune_way:    
            probe_out_dim_metric = probe_out_dim_metric.reshape(num_heads, -1)
            # Sum over the last dimension and take absolute values
            summed_metrics = torch.abs(probe_out_dim_metric.sum(dim=-1))
            # Sort the summed metrics
            sorted_value, sorted_indices = torch.sort(summed_metrics, dim=0)
            # print('summed_metricssorted_value', sorted_value)
            # Determine the number of heads to prune
            if 'pq' in cfg['prune_method']:
                num_prune_heads = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma, f'{self.key}_{prune_module}')[0]
            else:
                num_prune_heads = int(self.prune_hyper * num_heads)
                num_prune_heads = nearest_multiple(head_dim * num_prune_heads, probe_out_dim_metric.numel(), multiple, head_dim) // head_dim
            # Select the heads to prune
            heads_to_preserve = sorted_indices[num_prune_heads:]
            full_indices_to_preserve = (torch.arange(head_dim, device=probe_out_dim_metric.device) + heads_to_preserve.unsqueeze(1) * head_dim).view(-1)
            num_heads = num_heads - num_prune_heads
            return full_indices_to_preserve, None, num_heads, head_dim
            return full_indices_to_preserve, None, None, num_heads, head_dim
        elif 'each' in prune_way:
            probe_out_dim_metric = probe_out_dim_metric.reshape(num_heads, -1)
            # Sort the probe_out_dim_metric across each head
            sorted_value, sorted_indices = torch.sort(probe_out_dim_metric, dim=1)

            # Select indices to prune for each head
            
                # Find the indices where the condition is true
                # indices = torch.where(sorted_probe_temp == value_to_find)

                # # indices will be a tuple where the first element contains the indices of the matching elements
                # # Since you're interested in the first occurrence, you can take the first element of the first item in the tuple
                # if indices[0].nelement() != 0:  # Check if there's at least one match
                #     index = indices[0][0].item()
                #     print('Index:', index)
                # else:
                #     print('Value not found in tensor')
            if 'pq' in cfg['prune_method']:
                num_prune = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma, f'{self.key}_{prune_module}')[0]
            else:
           
                # Determine the number of elements to prune in each head
                # handling the edge case for RoPE if prune each head for qk
                num_prune_head_dim = int(self.prune_hyper * head_dim)
                num_prune_head_dim = nearest_multiple(num_prune_head_dim * num_heads, probe_out_dim_metric.numel(), multiple, num_heads) // num_heads
                indices_to_preserve = sorted_indices[:, num_prune_head_dim:]
            # Generate a range tensor for head indices
            head_range = torch.arange(num_heads, device=probe_out_dim_metric.device) * head_dim
            # Create the full indices for pruning using broadcasting
            full_indices_to_preserve = (indices_to_preserve + head_range.unsqueeze(1)).view(-1)

            head_dim = head_dim - num_prune_head_dim

            # return full_indices_to_preserve, indices_to_preserve, None, num_heads, head_dim
            return full_indices_to_preserve, indices_to_preserve,  num_heads, head_dim
        elif 'fill' in prune_way:
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
            

            if 'pq' in cfg['prune_method']:
                num_prune = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma, f'{self.key}_{prune_module}')[0]
            else:
                num_prune = int(probe_out_dim_metric.shape[0] * self.prune_hyper)
                num_prune = nearest_multiple(num_prune, probe_out_dim_metric.shape[0], multiple)
                print(f'{self.key}_{prune_module} num_prune', num_prune)
                threshold = sorted_value[num_prune]
                print(f'{self.key}_{prune_module} threshold', threshold)
                probe_out_dim_metric = probe_out_dim_metric.reshape(num_heads, -1)
                temp_sorted_value, temp_sorted_indices = torch.sort(probe_out_dim_metric, dim=1)
                print(f'{self.key}_{prune_module} temp_sorted_value', temp_sorted_value)
                elements_le_threshold = temp_sorted_value <= threshold
                print(f'{self.key}_{prune_module} elements_le_threshold', elements_le_threshold)
                # Count how many elements in each row are <= threshold
                count_per_row = elements_le_threshold.sum(dim=1)
                print(f'{self.key}_{prune_module} count_per_row', count_per_row)
                # Count how many elements in each column are <= threshold
                count_per_column = elements_le_threshold.sum(dim=0)
                print(f'{self.key}_{prune_module} count_per_column', count_per_column)
                # Number of rows and columns with at least one element <= threshold
                num_rows_with_le_threshold = count_per_row == probe_out_dim_metric.shape[1]
                num_columns_with_le_threshold = count_per_column == probe_out_dim_metric.shape[0]
                
                print(f'Number of rows with elements <= threshold: {num_rows_with_le_threshold.sum().item()}')
                print(f'Number of columns with elements <= threshold: {num_columns_with_le_threshold.sum().item()}')
                
            
            print(f'{self.key}_{prune_module} num_prune', num_prune)
            return sorted_indices[num_prune:], None, None, num_heads, head_dim
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
            sum_squared_norms = torch.sum(torch.linalg.vector_norm(h, ord=2, dim=1) ** 2, dim=0)

            average_squared_norm = sum_squared_norms / torch.tensor(bsz, device=h.device, dtype=torch.float)

            # Now compute norm_across_other_dims using scaler_inp
            # Assuming layer_info['weight'] is defined and has the appropriate shape
            norm_across_other_dims = (torch.sqrt(average_squared_norm.unsqueeze_(0).reshape((1,-1))) * torch.abs(layer_info['weight'])).sum(dim=0)

            # second piece
            # nsamples = 0
            # for i in range(bsz):
            #     scaler_inp *= nsamples / (nsamples + 1)
            #     scaler_inp += torch.linalg.vector_norm(h[i], ord=2, dim=1) ** 2 / (nsamples + 1)
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
            sum_squared_norms = torch.sum(torch.linalg.vector_norm(h, ord=2, dim=1) ** 2, dim=0)

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
    #     # if 'probe' in cfg['prune_method']:
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
    #     # elif 'probe' in cfg['prune_method'] and cfg['prune_metric'] == 'WIFN':
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
    #     # if 'probe' in cfg['prune_method']:
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
#             data = sorted_value.cpu().numpy()
#             stat, p = shapiro(data)
#             print('Shapiro-Wilk Test: Statistics=%.3f, ord=%.3f' % (stat, p))

#             # Interpret
#             alpha = 0.05
#             if p > alpha:
#                 print('Sample looks Gaussian (fail to reject H0)')
#             else:
#                 print('Sample does not look Gaussian (reject H0)')


#             from scipy.stats import kstest, norm, laplace

# # Assuming 'data' is your raw data
#             # Note: No pre-normalization before these tests
#             # You can estimate the parameters (mean and std for Gaussian, location and scale for Laplace) from your data if needed.

#             # KS test against Gaussian distribution, using data as-is or with estimated parameters
#             stat, p = kstest(data, 'norm', args=(data.mean(), data.std()))
#             print(f'KS Test against Gaussian: stat={stat:.3f}, ord={p:.3f}')

#             # KS test against Laplace distribution, using data as-is or with estimated parameters
#             # For Laplace, parameters are location (median) and scale (diversity, which can be estimated)
#             location, scale = laplace.fit(data)  # Fit might be used to estimate parameters based on your data
#             stat_laplace, p_laplace = kstest(data, 'laplace', args=(location, scale))
#             print(f'KS Test against Laplace: stat={stat_laplace:.3f}, ord={p_laplace:.3f}')

#             from scipy.stats import kstest, gamma
#             a, loc, scale = gamma.fit(data)
#             print(f'Gamma distribution fit: a={a:.3f}, loc={loc:.3f}, scale={scale:.3f}')
#             # Perform the KS test against the Gamma distribution
#             stat, p = kstest(data, 'gamma', args=(a, loc, scale))

#             print(f'KS Test against Gamma: stat={stat:.3f}, ord={p:.3f}')


#             from scipy.stats import kstest, lognorm

#             # Fit parameters from data
#             shape, loc, scale = lognorm.fit(data, floc=0)  # Force location parameter to zero for fitting

#             # Perform KS test against fitted log-normal distribution
#             stat, p = kstest(data, 'lognorm', args=(shape, loc, scale))
#             print(f'KS Test against Log-Normal: stat={stat:.3f}, p-value={p:.3f}')
#                         # print('\nstdddddddd')
#             import scipy.stats as stats

#             # Assuming 'data' is your layer's data
#             skewness = stats.skew(data)
#             kurtosis = stats.kurtosis(data, fisher=False)  # Set fisher=False to get kurtosis in comparison to normal distribution (which is 3)

#             print(f"Skewness: {skewness}")
#             print(f"Kurtosis: {kurtosis}")
#             # sorted_value = (sorted_value - torch.mean(sorted_value)) / torch.std(sorted_value) 
#             # print(f'{self.key} standardlized_value', sorted_value)
#             # # ensure value > 0
#             # sorted_value = (sorted_value - torch.min(sorted_value))
     
#             # # sorted_value, best_lambda = stats.boxcox(sorted_value)
#             # # print(f'{self.key} box_cox_transformation', sorted_value)
#             # # if high_outliers_num > 0:  # Check to ensure we have outliers to exclude
#             # #     sorted_value = sorted_value[: -high_outliers_num]
#             # # else:
#             # #     adjusted_sorted_value = sorted_value
#             # num_prune = cal_prune_count_base_on_pq(sorted_value, self.pq_p, self.pq_q, self.eta, self.pq_beta, self.pq_gamma, self.key)[0]
#             # nominator_varying_vector_norm, denominator_varying_vector_norm, dimension = parallel_cal_varying_length_info(sorted_value, 1, 0.5)
#             # # print('nominator_varying_vector_norm', nominator_varying_vector_norm.shape, nominator_varying_vector_norm)
#             # # print('denominator_varying_vector_norm', denominator_varying_vector_norm.shape, denominator_varying_vector_norm)
#             # ratio = (nominator_varying_vector_norm / denominator_varying_vector_norm)
#             # print(f'{self.key}_ratio', ratio)
#                # print(f'{self.key} standardlized_value after adding', sorted_value)
#             # Choose a lambda for the Box-Cox transformation
#             # lam = 0.5  # Example lambda value, you might need to adjust this based on your data
#             # total_sum = sorted_value.sum()
#             # sorted_value = sorted_value / total_sum
#             # # Apply the Box-Cox transformation
#             # sorted_value = self.box_cox_transformation(sorted_value, -2)
#             # sorted_value_np = sorted_value.cpu().numpy()

#             # # Apply the Box-Cox transformation
#             # transformed_value, best_lambda = stats.boxcox(sorted_value_np)

#             # # Optionally, convert the transformed data back to a PyTorch tensor
#             # sorted_value = torch.from_numpy(transformed_value).to(probe_out_dim_metric.device)

#             import numpy as np
#             import matplotlib.pyplot as plt
#             from scipy.signal import find_peaks
#             from scipy.interpolate import interp1d

#             def calculate_FWHM(data, bins=100, plot=False):
#                 # Create a histogram of the data
#                 counts, bin_edges = np.histogram(data, bins=bins, density=True)
#                 bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
#                 # Find the peak of the histogram
#                 peak_idx = np.argmax(counts)
#                 peak_value = bin_centers[peak_idx]
#                 half_max = counts[peak_idx] / 2
                
#                 # Interpolate to find the FWHM
#                 interp = interp1d(bin_centers, counts - half_max, kind='cubic')
#                 roots = interp1d(bin_centers, counts - half_max, kind='cubic').roots()
                
#                 # Find the roots closest to the peak to define the FWHM
#                 valid_roots = roots[(roots > bin_centers[0]) & (roots < bin_centers[-1])]
#                 left_root = valid_roots[valid_roots < peak_value][-1]
#                 right_root = valid_roots[valid_roots > peak_value][0]
                
#                 fwhm = right_root - left_root
                
#                 if plot:
#                     plt.plot(bin_centers, counts, label='Data Histogram')
#                     plt.plot([left_root, right_root], [half_max, half_max], 'ro-')
#                     plt.title('Histogram and FWHM')
#                     plt.legend()
#                     plt.show()
                
#                 return fwhm

#             # Assuming 'data' is your dataset
#             data = sorted_value.cpu().numpy()  # Example: Converting from PyTorch tensor to NumPy array
#             fwhm = calculate_FWHM(data, plot=False)

            # print(f"FWHM: {fwhm}")
            # print(f'{self.key} box_cox_transformation', transformed_value_tensor)
        # let the remaining element be the multiple of multiple to fit tensor cores