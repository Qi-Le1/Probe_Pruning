import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from config import cfg
import copy

def cal_pq_index(tensor, norm, norm_dim_index, dimension):
    shape_temp = tensor.shape
    # set p and q
    p = 1
    q = 2
    norm_p = torch.norm(norm, p=p, dim=norm_dim_index)
    norm_q = torch.norm(norm, p=q, dim=norm_dim_index) + 1e-10
    
    # set dimension based on tensor dimensions
    # dimension = tensor.shape[dimension_index]

    # compute PQ_index for all samples at once
    # first dimension will be the batch size
    PQ_index_list = (1 - dimension ** (1/q - 1/p) * norm_p / norm_q)
    # c = PQ_index_list.shape
    # check for NaN values
    if PQ_index_list.dim() == 0:
        PQ_index_list = PQ_index_list.unsqueeze(0)

    if torch.isnan(PQ_index_list).any():
        raise ValueError('PQ_index_list contains nan values')

    eta = cfg['delete_threshold']
    if eta == 999:
        return tensor, None
    gamma = 1
    beta = 0.9
    # beta = 
    lower_bound = dimension * (1 + eta) ** (-q / (q - p)) * (1 - PQ_index_list) ** (q * p / (q - p))
    beta_tensor = torch.full_like(lower_bound, beta)
    number_of_pruned_channel = torch.floor(dimension * torch.min(gamma * (1 - lower_bound / dimension) , beta_tensor))
    # PQ_index = copy.deepcopy(PQ_index_list[0]).item()
    return PQ_index_list, number_of_pruned_channel

def get_scalar_value(x):
    if isinstance(x, torch.Tensor):
        if x.dim() == 0 or (x.dim() == 1 and x.numel() == 1):
            return x.item()
    return x

class ChannelDeletor():

    def __init__(self):
        # self.zero_channel = 0
        # self.zero_channel_for_batch = 0
        # self.zero_single_channel_for_batch = 0
        # self.total_batch_channel = 0
        # self.sample_channel = 0
        # self.sparsity_ratio = 0
        # self.sparsity_num = 0

        # self.PQ_index_num = 0
        # self.PQ_index_list_distribution_mean = None
        return  
    
    def add_info_to_logger(self, logger):

        return

    def calculate_empty_channel(self, tensor):
        tensor = tensor.clone().cpu()

        tensor_zero_mask = tensor < 1e-5

        zeros_count_per_sample_channel = torch.sum(tensor_zero_mask, dim=(2, 3)) if tensor.dim() > 2 else tensor_zero_mask

        # if zeros count equals dimension, then this whole channel is empty 
        if tensor.dim() > 2:
            dimension = tensor.shape[2] * tensor.shape[3]
        else:
            dimension = tensor.shape[0]

        is_channel_empty_per_sample = zeros_count_per_sample_channel == dimension

        # per sample
        empty_channel_count_per_sample = is_channel_empty_per_sample.sum(dim=1)

        # per channel in the batch
        self.samples_with_empty_channel_count_per_batch = is_channel_empty_per_sample.sum(dim=0)

        # per channel for all batch samples
        # print('--', tensor_zero_mask.sum(dim=0), (tensor_zero_mask.sum(dim=0) == tensor.shape[0]))
        empty_whole_channel_count_in_all_samples = (tensor_zero_mask.sum(dim=0) == tensor.shape[0]).sum()

        total_channel_count_per_sample = tensor.shape[1]
        total_channel_count_in_all_samples = tensor.shape[0] * tensor.shape[1]
        
        self.empty_single_channel_count_in_all_samples = is_channel_empty_per_sample.sum().item()
        # self.zero_single_channel_for_batch = zero_single_channel_for_batch
        self.empty_whole_channel_count_in_all_samples = empty_whole_channel_count_in_all_samples.item()

        self.total_channel_count_per_sample = total_channel_count_per_sample
        self.total_channel_count_in_all_samples = total_channel_count_in_all_samples
        
        self.empty_single_channel_count_in_all_samples_ratio = self.empty_single_channel_count_in_all_samples / self.total_channel_count_in_all_samples
        self.empty_whole_channel_count_in_all_samples_ratio = self.empty_whole_channel_count_in_all_samples / self.total_channel_count_per_sample
        # print(self.empty_whole_channel_count_in_all_samples, self.total_channel_count_per_sample, self.empty_whole_channel_count_in_all_samples_ratio, 'ssss')
        zero_activations = torch.sum(tensor == 0)
        total_activations = tensor.numel()
        self.sparsity_ratio = zero_activations.item() / total_activations
        return 


    def delete_channel_by_sparsity(self, tensor):
        '''
        larger sparsity means a sparser vector
        '''
        self.calculate_empty_channel(tensor)

        select_channel = torch.arange(tensor.shape[1])

        self.samples_with_empty_channel_count_per_batch_ratio = np.array(self.samples_with_empty_channel_count_per_batch / tensor.shape[0])
        # samples_with_non_empty_channel_count_per_batch_ratio = 1 - self.samples_with_empty_channel_count_per_batch_ratio
        mask = samples_with_empty_channel_count_per_batch_ratio < cfg['delete_threshold']
        select_channel = select_channel[mask]

        delete_channel_mask = samples_with_empty_channel_count_per_batch_ratio >= cfg['delete_threshold']

        self.delete_channel_ratio = delete_channel_mask.sum() / tensor.shape[1]
        if tensor.dim() > 2:
            tensor[:, delete_channel_mask, :, :] = 0
        else:
            tensor[:, delete_channel_mask] = 0
        return tensor, select_channel
    
    # def cal_PQ_index(self, tensor):
       
    #     # update PQ_index_num
    #     self.PQ_index_num += tensor.shape[1]

    #     return number_of_pruned_channel

    def delete_channel_by_PQ_index(self, tensor):
        # self.calculate_empty_channel(tensor)

        '''
        smaller PQ_index means a denser vector  
        larger PQ_index means a sparser vector
        '''
        # tensor = tensor.clone().cpu()
        # set p and q
        p = 1
        q = 2

        # flatten tensor along the channel dimension
        # flat_tensor = tensor.view(tensor.shape[1], -1)

        # Frobenius norm
        if tensor.ndimension() == 4:
            norm = torch.norm(tensor, p=cfg['prune_norm'], dim=(2, 3))
        elif tensor.ndimension() == 3:
            norm = torch.norm(tensor, p=cfg['prune_norm'], dim=(2))
        elif tensor.ndimension() == 2:
            norm = torch.pow(tensor, cfg['prune_norm'])
        
        # compute norms
        # norm_p = torch.norm(norm_f, p=p, dim=1)
        # norm_q = torch.norm(norm_f, p=q, dim=1) + 1e-10
        
        # # set dimension based on tensor dimensions
        # dimension = tensor.shape[1]

        # # compute PQ_index for all samples at once
        # PQ_index_list = (1 - dimension ** (1/q - 1/p) * norm_p / norm_q)
        # self.PQ_index = copy.deepcopy(PQ_index_list[0]).item()
        # print('pq', self.PQ_index)
        # # c = PQ_index_list.shape
        # # check for NaN values
        # if any(math.isnan(val) for val in PQ_index_list):
        #     raise ValueError('PQ_index_list contains nan values')

        # eta = cfg['delete_threshold']
        # if eta == 999:
        #     return tensor, None
        # gamma = 1
        # beta = 0.9
        # # beta = 
        # lower_bound = dimension * (1 + eta) ** (-q / (q - p)) * (1 - PQ_index_list) ** (q * p / (q - p))
        # beta_tensor = torch.full_like(lower_bound, beta)
        # number_of_pruned_channel = torch.floor(dimension * torch.min(gamma * (1 - lower_bound / dimension) , beta_tensor))
        
        PQ_index_list, number_of_pruned_channel = cal_pq_index(tensor, norm, 1, tensor.shape[1])
        if number_of_pruned_channel == None:
            return tensor, None
        
        # Compute the mean along the 0th dimension
        self.PQ_index = torch.mean(PQ_index_list, dim=0).item()

        self.number_of_pruned_channel = torch.mean(number_of_pruned_channel, dim=0).item()
        self.delete_channel_ratio = number_of_pruned_channel.sum().item() / tensor.shape[1]
        # print('delete_channel_ratio', self.delete_channel_ratio)
        # print(gamma * (1 - lower_bound / dimension) , beta_tensor, number_of_pruned_channel, tensor.shape[1])
        # self.PQ_index_list_distribution = cur_PQ_index_list

        # if self.PQ_index_list_distribution_mean is None:
        #     self.PQ_index_list_distribution_mean = copy.deepcopy(cur_PQ_index_list)
        # else:
        #     self.PQ_index_list_distribution_mean = ((self.PQ_index_num - len(cur_PQ_index_list)) * self.PQ_index_list_distribution_mean + len(cur_PQ_index_list) * cur_PQ_index_list) / self.PQ_index_num
        
        # Sort norm_f along dim=1 (channels dimension) and get indices
        sorted_norms, indices = torch.sort(norm, dim=1)

        # Select number_of_pruned_channel smallest values
        if cfg['server']['batch_size']['test'] == 1:
            for i in range(norm.size(0)):  # iterate over the first dimension (n)
                indices_to_prune = indices[i, :int(number_of_pruned_channel.item())]  # get the indices to prune

                if tensor.dim() > 2:
                    tensor[i, indices_to_prune, :, :] = 0
                else:
                    tensor[i, indices_to_prune] = 0

        # shrink the whole column
        if cfg['server']['batch_size']['test'] != 1:
            if cfg['batch_deletion'] == 'PQ':
                if tensor.ndimension() == 4:
                    norm = torch.norm(tensor, p=cfg['prune_norm'], dim=(0, 2, 3))
                elif tensor.ndimension() == 3:
                    norm = torch.norm(tensor, p=cfg['prune_norm'], dim=(0, 2))
                elif tensor.ndimension() == 2:
                    norm = torch.norm(tensor, p=cfg['prune_norm'], dim=(0))

                PQ_index_list, number_of_pruned_channel = cal_pq_index(tensor, norm, 0, tensor.shape[1])

                self.PQ_index = torch.mean(PQ_index_list, dim=0).item()

                self.number_of_pruned_channel = number_of_pruned_channel.item()
                self.delete_channel_ratio = number_of_pruned_channel.sum().item() / tensor.shape[1]

                sorted_norms, indices = torch.sort(norm, dim=0)

                if tensor.dim() > 2:
                    tensor[:, indices[:int(number_of_pruned_channel.item())], :, :] = 0
                else:
                    tensor[:, indices[:int(number_of_pruned_channel.item())]] = 0

            elif cfg['batch_deletion'] == 'inter':
                
                indices_to_prune = [[] for i in range(tensor.shape[0])]
                for i in range(tensor.shape[0]):  # iterate over the first dimension (n)
                    indices_to_prune[i] = indices[i, :int(number_of_pruned_channel[i].item())]  # get the indices to prune

                # # Initialize the intersection set with the first row
                # intersection_set_tensor = set(indices_to_prune[0].numpy())

                # # Compute the intersection across all rows
                # for row in indices_to_prune[1:]:
                #     intersection_set_tensor &= set(row.numpy())

                # Initialize the intersection set with the first tensor row
                intersection_tensor = indices_to_prune[0]

                # Compute the intersection across all rows
                for row in indices_to_prune[1:]:
                    intersection_tensor = torch.tensor(list(set(intersection_tensor.tolist()) & set(row.tolist())))

                print('intersection_set_tensor', len(intersection_tensor), tensor.shape[1])
                self.delete_channel_ratio = len(intersection_tensor) / tensor.shape[1]

                if tensor.dim() > 2:
                    tensor[:, list(intersection_tensor), :, :] = 0
                else:
                    tensor[:, list(intersection_tensor)] = 0

        select_channel = None
        return tensor, select_channel
        
    def delete_channel(self, tensor):
        # if cfg['delete_criteria'] == 'NA':
        #     return tensor, None
        if cfg['delete_criteria'] == 'sparsity':
            return self.delete_channel_by_sparsity(tensor)
        elif cfg['delete_criteria'] == 'PQ':
            return self.delete_channel_by_PQ_index(tensor)
        else:
            return tensor, None

class CustomReLU(nn.Module):

    def __init__(self, threshold=0):
        super().__init__()
        self.relu_threshold = threshold
        self.channel_deletor = ChannelDeletor()
        return

    def forward(self, x):
        out = torch.where(x >= self.relu_threshold, x, torch.tensor(0, dtype=x.dtype, device=x.device))
        selected_channel = None
        if cfg['delete_method'] == 'our':
            out, selected_channel = self.channel_deletor.delete_channel(out)
        return out, selected_channel

class ParameterDeletor():

    def __init__(self, layer_type):
        # super().__init__()
        if isinstance(layer_type, nn.Conv2d):
            self.layer_type = 'conv'
        elif isinstance(layer_type, nn.Linear):
            self.layer_type = 'linear'
        else:
            print('layer_type not correct', layer_type)

        self.pruned_dimension = None

        return

    def prune_weights_by_magnitude(self, module, norm, number_of_pruned_eles):
        # Flatten the weights and get their absolute values
        flat_weights = module.weight.data.view(-1)

        norm = norm.view(-1)
        # Get the indices of the smallest weights by magnitude
        _, indices_to_prune = torch.topk(norm, get_scalar_value(number_of_pruned_eles), largest=False)

        # Set the weights corresponding to these indices to zero
        flat_weights[indices_to_prune] = 0

        module.weight.data = flat_weights.view(module.weight.data.shape)

    def unstructured(self, module):
        if self.layer_type in 'conv':
            
            if cfg['delete_criteria'] == 'PQ':
                copy_weight = module.weight.data.view(-1).clone()
                norm = torch.pow(copy_weight, cfg['prune_norm'])
                total_elements = torch.prod(torch.tensor(module.weight.data.shape)).item()
                PQ_index_list, number_of_pruned_eles = cal_pq_index(copy_weight, norm, 0, total_elements)
                if number_of_pruned_eles == None:
                    return
                # prune.l1_unstructured(module, name='weight', amount=cfg['delete_threshold'])
                # prune.remove(module, 'weight')
                self.prune_weights_by_magnitude(module, norm, number_of_pruned_eles)
                
                self.delete_channel_ratio = number_of_pruned_eles.sum().item() / total_elements
        elif self.layer_type == 'linear':
            pass
        return
    
    def get_indices_to_prune(self, sorted_indices, number_of_pruned_channel):
        indices_to_prune = sorted_indices[:int(number_of_pruned_channel[0].item())]  # get the indices to prune
        return indices_to_prune


    def set_weight_to_zero(self, tensor, indices_to_prune):
        # indices_to_prune = indices[:int(number_of_pruned_channel[0].item())]  # get the indices to prune

        if self.pruned_dimension == 0:
            if tensor.dim() == 4:
                tensor[indices_to_prune, :, :, :] = 0
            else:
                tensor[indices_to_prune, :] = 0
        elif self.pruned_dimension == 1:
            if tensor.dim() == 4:
                tensor[:, indices_to_prune, :, :] = 0
                print('channel-wise prune index', len(indices_to_prune), indices_to_prune)
            else:
                tensor[:, indices_to_prune] = 0

        return
    
    def channel_wise(self, module):
        if self.layer_type == 'conv':
            # Compute L1 norm for each channel
            # out_channels, in_channels, kernel_height, kernel_width
            norm = torch.norm(module.weight.data, p=cfg['prune_norm'], dim=(0, 2, 3))
        
            if cfg['delete_criteria'] == 'PQ':
                sorted_norms, sorted_indices = torch.sort(norm, dim=-1)
                PQ_index_list, number_of_pruned_channel = cal_pq_index(module.weight.data, norm, 0, module.weight.data.shape[1])
                if number_of_pruned_channel == None:
                    return
                indices_to_prune = self.get_indices_to_prune(sorted_indices, number_of_pruned_channel)
                self.set_weight_to_zero(module.weight.data, indices_to_prune)
                self.delete_channel_ratio = number_of_pruned_channel.sum().item() / module.weight.data.shape[1]

                print('channel-wise prune ratio', self.delete_channel_ratio)
        elif self.layer_type == 'linear':
            # Similar approach for Linear layers
            pass
        return indices_to_prune
    
    def filter_wise(self, module):
        if self.layer_type == 'conv':
            norm = torch.norm(module.weight.data, p=cfg['prune_norm'], dim=(1, 2, 3))
        
            if cfg['delete_criteria'] == 'PQ':
                sorted_norms, sorted_indices = torch.sort(norm, dim=-1)
                PQ_index_list, number_of_pruned_filter = cal_pq_index(module.weight.data, norm, 0, module.weight.data.shape[0])
                if number_of_pruned_filter == None:
                    return
                indices_to_prune = self.get_indices_to_prune(sorted_indices, number_of_pruned_filter)
                self.set_weight_to_zero(module.weight.data, indices_to_prune)
                self.delete_channel_ratio = number_of_pruned_filter.sum().item() / module.weight.data.shape[0]
        elif self.layer_type == 'linear':
            # Similar approach for Linear layers
            pass
        return indices_to_prune
    

    def process(self, module):
        if cfg['delete_method'] == 'unstructured':
            self.unstructured(module)
        elif cfg['delete_method'] == 'channel-wise':
            indices_to_prune = self.channel_wise(module)
            self.pruned_dimension = 1
        elif cfg['delete_method'] == 'filter-wise':
            indices_to_prune = self.filter_wise(module)
            self.pruned_dimension = 0
        else:
            print('no parameter deletion\n')

        return indices_to_prune


class InferenceConv2d(nn.Module):

    def __init__(self, conv_layer):
        super().__init__()
        
        self.conv = conv_layer
        self.parameter_deletor = ParameterDeletor(conv_layer)
        self.indices_to_prune = self.parameter_deletor.process(self.conv)
        # first dimension becomes the batch size
        self.indices_to_prune += 1
        return
        # self.weight = conv_layer.weight
        # self.bias = conv_layer.bias
        # self.stride = conv_layer.stride
        # self.padding = conv_layer.padding
        # self.dilation = conv_layer.dilation
        # self.groups = conv_layer.groups

    def prune_input_based
    def forward(self, x, selected_channel=None):
        
        
        # if selected_channel is not None:
        #     weight = self.weight[:, selected_channel, :, :]
        # else:
        #     weight = self.weight
        # weight = self.weight
        # out = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        out = self.conv(x)
        return out


class InferenceLinear(nn.Module):

    def __init__(self, linear_layer):
        super().__init__()
        # self.weight = linear_layer.weight
        # self.bias = linear_layer.bias
        self.linear = linear_layer
        self.parameter_deletor = ParameterDeletor(linear_layer)
        indices_to_prune = self.parameter_deletor.process(self.linear)
        return
    
    def forward(self, x, selected_channel=None):

        # if selected_channel is not None:
        #     weight = self.weight[:, selected_channel]
        # else:
        #     weight = self.weight
        # weight = self.weight
        # out = F.linear(x, weight, self.bias)

        out = self.linear(x)
        return out
    

# class CustomReLU(nn.Module):
#     def __init__(self, threshold=0):
#         super().__init__()
#         self.relu_threshold = threshold
#         self.layer_sparsity = {}
#         self.set_column_to_zero = False
#         self.batch_size = cfg['server']['batch_size']['test']
#         self.PQ_index_num = 0

#     def hook_fn(self, module, input, output):
#         """
#         Generate a hook function with a specific threshold
#         """

#         # set_column_to_zero(output) if self.set_column_to_zero else None
        
#         zero_activations = torch.sum(output == 0)
#         total_activations = output.numel()
#         sparsity = zero_activations.item() / total_activations
#         # print(output.shape)
#         if self.batch_size not in self.layer_sparsity:
#             self.layer_sparsity[self.batch_size] = {}

#         if self.relu_threshold not in self.layer_sparsity[self.batch_size]:
#             self.layer_sparsity[self.batch_size][self.relu_threshold] = {
#                 'total_sparsity_ratio': [], 
#                 'sparsity_ratio': 0,
#                 'empty_single_channel': 0,
#                 'total_batch_channel': 0,
#                 'empty_all_channel': 0,
#                 'sample_channel': 0,
#                 'PQ_index_list': [],
#                 'PQ_index_list_distribution_mean': [],
#                 'PQ_index_list_mean': []
#             }

#         self.calculate_empty_channel(output)
#         self.remove_column(output)
#         self.layer_sparsity[self.batch_size][self.relu_threshold]['total_sparsity_ratio'].append(sparsity)
#         layer_sparsity_ratio_sum = sum(self.layer_sparsity[self.batch_size][self.relu_threshold]['total_sparsity_ratio'])
#         layer_sparsity_ratio_len = len(self.layer_sparsity[self.batch_size][self.relu_threshold]['total_sparsity_ratio'])
#         self.layer_sparsity[self.batch_size][self.relu_threshold]['sparsity_ratio'] = layer_sparsity_ratio_sum / layer_sparsity_ratio_len
#         self.layer_sparsity[self.batch_size][self.relu_threshold]['PQ_index_list_mean'] = np.mean(self.layer_sparsity[self.batch_size][self.relu_threshold]['PQ_index_list_distribution_mean'])
#         return 

#     def register_hooks(self, set_column_to_zero=False):
#         """
#         Register the forward hook to all the modules in the model
#         """
#         self.register_forward_hook(self.hook_fn)
#         if set_column_to_zero:
#             self.set_column_to_zero = True

#     def forward(self, x):
#         return torch.where(x >= self.relu_threshold, x, torch.tensor(0, dtype=x.dtype, device=x.device))
#         # return F.relu(x - self.threshold) + self.threshold

#     def calculate_empty_channel(self, tensor):
        
#         # for c in range(tensor.shape[1]):
#         #     cur_batch_all_channel_zero = True
#         #     for sample in range(tensor.shape[0]):
#         #         zero_counts = 0
#         #         if tensor.dim() > 2:
#         #             zero_counts = torch.sum(tensor[sample, c, :, :] == 0).item()
#         #             # cur_tensor = copy.deepcopy(tensor[sample, c, :, :])
#         #         else:
#         #             zero_counts = torch.sum(tensor[sample, c] == 0).item()
#         #             # cur_tensor = copy.deepcopy(tensor[sample, c])
#         #         if zero_counts == 0:
#         #             self.layer_sparsity[self.batch_size][self.relu_threshold]['empty_single_channel'] += 1
#         #         else:
#         #             cur_batch_all_channel_zero = False
#         #         self.layer_sparsity[self.batch_size][self.relu_threshold]['total_batch_channel'] += 1
#         #     if cur_batch_all_channel_zero:
#         #         self.layer_sparsity[self.batch_size][self.relu_threshold]['empty_all_channel'] += 1
#         #     self.layer_sparsity[self.batch_size][self.relu_threshold]['sample_channel'] += 1

#         # clone and move tensor to CPU
#         tensor = tensor.clone().cpu()

#         tensor_zero_mask = tensor == 0

#         # calculate single channel zero counts
#         single_channel_zero_counts = torch.sum(tensor_zero_mask, dim=(0,2,3)) if tensor.dim() > 2 else torch.sum(tensor_zero_mask, dim=0)
#         empty_single_channel_count = torch.sum(single_channel_zero_counts == tensor[0].numel()).item()
#         self.layer_sparsity[self.batch_size][self.relu_threshold]['empty_single_channel'] += empty_single_channel_count
#         self.layer_sparsity[self.batch_size][self.relu_threshold]['total_batch_channel'] += tensor.shape[0] * tensor.shape[1]

#         # calculate all channel zero counts
#         all_channel_zero_counts = torch.sum(single_channel_zero_counts == tensor.shape[0])
#         self.layer_sparsity[self.batch_size][self.relu_threshold]['empty_all_channel'] += all_channel_zero_counts.item()
#         self.layer_sparsity[self.batch_size][self.relu_threshold]['sample_channel'] += tensor.shape[1]

#         return

#     def cal_PQ_index(self, tensor):

#         # tensor = tensor.clone().cpu()
#         # p = 1
#         # q = 2
        
#         # if tensor.dim() > 2:
#         #     dimension = tensor.shape[0] * tensor.shape[2] * tensor.shape[3]
#         # else:
#         #     dimension = tensor.shape[0]

#         # PQ_index_list = []
#         # for i in range(tensor.shape[1]):
#         #     if tensor.dim() > 2:
#         #         cur_tensor = copy.deepcopy(tensor[:, i, :, :])
#         #     else:
#         #         cur_tensor = copy.deepcopy(tensor[:, i])
#         #     PQ_index = 1 - dimension ** (1/q-1/p) * np.linalg.norm(cur_tensor.flatten(), ord=p) / (np.linalg.norm(cur_tensor.flatten(), ord=q) + 1e-10)
#         #     # print(dimension ** (1/q-1/p), np.linalg.norm(cur_tensor.flatten(), ord=p), (np.linalg.norm(cur_tensor.flatten(), ord=q) + 1e-10), np.linalg.norm(cur_tensor.flatten(), ord=p) / (np.linalg.norm(cur_tensor.flatten(), ord=q) + 1e-10),  dimension ** (1/q-1/p) * np.linalg.norm(cur_tensor.flatten(), ord=p) / (np.linalg.norm(cur_tensor.flatten(), ord=q) + 1e-10))
#         #     if np.isnan(PQ_index):
#         #         raise ValueError('PQ_index is nan')
#         #     PQ_index_list.append(PQ_index)

#         #     self.PQ_index_num += 1

#         # return PQ_index_list

#         # clone and move tensor to CPU
#         tensor = tensor.clone().cpu()

#         # set p and q
#         p = 1
#         q = 2

#         # set dimension based on tensor dimensions
#         dimension = tensor.shape[0] * (tensor.shape[2] * tensor.shape[3] if tensor.dim() > 2 else 1)

#         # flatten tensor along the channel dimension
#         flat_tensor = tensor.view(tensor.shape[1], -1)

#         # compute norms
#         norm_p = torch.norm(flat_tensor, p=p, dim=1)
#         norm_q = torch.norm(flat_tensor, p=q, dim=1) + 1e-10

#         # compute PQ_index for all channels at once
#         PQ_index_list = (1 - dimension ** (1/q-1/p) * norm_p / norm_q).tolist()

#         # check for NaN values
#         if any(math.isnan(val) for val in PQ_index_list):
#             raise ValueError('PQ_index_list contains nan values')

#         # update PQ_index_num
#         self.PQ_index_num += tensor.shape[1]

#         return PQ_index_list

#     def remove_column(self, tensor):
        
#         cur_PQ_index_list = self.cal_PQ_index(tensor)
#         if len(self.layer_sparsity[self.batch_size][self.relu_threshold]['PQ_index_list_distribution_mean']) == 0:
#             self.layer_sparsity[self.batch_size][self.relu_threshold]['PQ_index_list_distribution_mean'] = copy.deepcopy(cur_PQ_index_list)
        
#         PQ_index_list_distribution_mean = self.layer_sparsity[self.batch_size][self.relu_threshold]['PQ_index_list_distribution_mean']

#         PQ_index_list_distribution_mean = np.array(PQ_index_list_distribution_mean)
#         cur_PQ_index_list = np.array(cur_PQ_index_list)


#         self.layer_sparsity[self.batch_size][self.relu_threshold]['PQ_index_list_distribution_mean'] = ((self.PQ_index_num - len(cur_PQ_index_list)) * PQ_index_list_distribution_mean + len(cur_PQ_index_list) * cur_PQ_index_list) / self.PQ_index_num

        
#         return


def init_param(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


def make_batchnorm(m, momentum, track_running_stats):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
            m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m


def loss_fn(output, target, reduction='mean'):
    if target.dtype == torch.int64:
        # print(f'cross_entropy: {output}, {target}')
        loss = F.cross_entropy(output, target, reduction=reduction)
        # if torch.isnan(loss) or loss.item() > 20:
        #     print(f'cross_entropy: {loss}')
        #     print(f'output: {output}\n')
        #     print(f'target: {target}\n')
    else:
        # print(f'mse: {output}, {target}')
        raise ValueError('not cross_entropy')
        loss = F.mse_loss(output, target, reduction=reduction)
        # print(f'mse_loss: {loss}')
    return loss


def mse_loss(output, target, weight=None):
    mse = F.mse_loss(output, target, reduction='none')
    mse = weight * mse if weight is not None else mse
    mse = torch.sum(mse)
    mse /= output.size(0)
    return mse


def cross_entropy_loss(output, target, weight=None):
    target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, reduction='mean', weight=weight)
    return ce


def kld_loss(output, target, weight=None, T=1):
    kld = F.kl_div(F.log_softmax(output, dim=-1), F.softmax(target / T, dim=-1), reduction='none')
    kld = weight * kld if weight is not None else kld
    kld = torch.sum(kld)
    kld /= output.size(0)
    return kld






