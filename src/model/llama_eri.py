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
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from transformers.pytorch_utils import Conv1D
from config import cfg
from math import prod
from .model import init_param, mse_loss
from typing import List, Optional, Tuple, Union
from module import to_device, TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING, TRANSFORMERS_MODELS_OUT_TARGET_MODULES_MAPPING
from functools import reduce

from .pruning_module import HiddenRepresentationPruning, WeightPruning

class LlamaEriModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.add_pruner(cfg['prune_name'])

    def add_pruner(self, prune_name):
        self._find_and_replace(prune_name)
        mark_no_trainable(self.model)
        if 'global' in cfg['prune_name'] and cfg['prune_tgt'] == 'weight':
            pruning_module = WeightPruning(cfg, 'global')
            pruning_module.global_pruning(self.model)
        
        return
    
    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

    def _create_new_module(self, prune_name, target, key):
        bias = hasattr(target, "bias") and target.bias is not None
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if cfg['prune_tgt'] == 'hidden_repr':
            in_features = getattr(target, 'in_features', None)
            out_features = getattr(target, 'out_features', None)
            pruning_module = HiddenRepresentationPruning(cfg, key,target.weight.device, in_features, out_features)
        elif cfg['prune_tgt'] == 'weight':
            pruning_module = WeightPruning(cfg, key)
        else:
            raise ValueError('Not valid prune tgt')
        
        kwargs = {
            "prune_tgt": cfg['prune_tgt'],
            "prune_metric": cfg['prune_metric'],
            "pruning_module": pruning_module,
            "key": key,
            "fan_in_fan_out": False,
            "dev": target.weight.device,
        }
        # if isinstance(target, torch.nn.Embedding):
        #     embedding_kwargs = kwargs.copy()
        #     embedding_kwargs.pop("fan_in_fan_out", None)
        #     in_features, out_features = target.num_embeddings, target.embedding_dim
        #     new_module = Embedding(prune_name, in_features, out_features, **embedding_kwargs)
        if isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            dilation = target.dilation
            groups = target.groups
            new_module = Conv2d(prune_name, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["fan_in_fan_out"] = True
                kwargs["is_target_conv_1d_layer"] = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = Linear(prune_name, in_features, out_features, bias=bias, **kwargs)

        return new_module

    def _find_and_replace(self, prune_name):
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key, module in self.model.named_modules():
            print('key: ', key, type(module), flush=True)
        # return
        target_modules = _get_target_modules(cfg)
        print('target_modules: ', target_modules)
        for key in key_list:
            if not _check_target_module_exists(target_modules, key):
                continue

            print('Replaced Layer Keys', key, flush=True)

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            
            # if 'WO' in cfg['prune_metric']:
            #     new_module.is_prune_out_dim = False
            #     out_dim_target_modules = TRANSFORMERS_MODELS_OUT_TARGET_MODULES_MAPPING[cfg['model_type']]
            #     if _check_target_module_exists(out_dim_target_modules, key):
            #         print('out_dim_target_modules', key)
            #         new_module.is_prune_out_dim = True

            new_module = self._create_new_module(prune_name, target, key)

            if 'WO' in cfg['prune_metric']:
                new_module.is_prune_out_dim = False
                out_dim_target_modules = TRANSFORMERS_MODELS_OUT_TARGET_MODULES_MAPPING[cfg['model_type']]
                if _check_target_module_exists(out_dim_target_modules, key):
                    print('out_dim_target_modules', key)
                    new_module.is_prune_out_dim = True

            self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        fan_in_fan_out = getattr(new_module, "fan_in_fan_out", False)
        # if fan_in_fan_out is True, the layer is conv1d layer in GPT2
        # which is a self-defined layer, not the traditional Conv1D
        new_module.weight = transpose(old_module.weight, fan_in_fan_out)
        new_module.weight.requires_grad = False
        new_module.device = old_module.weight.device
        new_module.is_pruned = True
        if hasattr(old_module, "bias"):
            # old_module might not have bias, bias=None
            # need to write into new_module, otherwise
            # the parent class will assign bias
            new_module.bias = old_module.bias

        if 'local' in cfg['prune_name'] and cfg['prune_tgt'] == 'weight':
            new_module.prune_weight(new_module.weight, new_module.layer_type)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

def transpose(weight, fan_in_fan_out):
    transposed_weight = weight.T if fan_in_fan_out else weight
    # return transposed_weight
    return nn.Parameter(transposed_weight)

def mark_no_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = False
    return

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def _get_target_modules(cfg):
    target_modules = cfg['cust_tgt_modules']
    return target_modules

# global_only_one_module = True
def _check_target_module_exists(target_modules, key):
    if isinstance(target_modules, str):
        target_module_found = re.fullmatch(target_modules, key)
    else:
        # target_module_found = any(key.endswith(target_key) for target_key in target_modules)
        target_module_found = any(key.endswith(target_key) for target_key in target_modules)

    # if target_module_found:
    #     global global_only_one_module
    #     if global_only_one_module:
    #         global_only_one_module = False
    #         return True
    #     else:
    #         return False
    return target_module_found

class EriLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.prune_tgt = kwargs['prune_tgt']
        self.prune_metric = kwargs['prune_metric']
        self.pruning_module = kwargs['pruning_module']
        self.key = kwargs['key']

        self.pruning_channel_ratio = []
        self.input_prune_channels = None
        self.weight_norm_across_channel_dims = None
        pass
    

    # TODO, weight.data
    def prune_weight(self, weight, layer_type):
        layer_info = {
            'weight': self.weight,
        }
        pruned_w, pruned_dims, prune_channels_multi_dims = self.pruning_module.local_pruning(weight, layer_type, layer_info, self.key)
        print('pruned_w.shape: ', pruned_w.shape, self.key)
        self.a = pruned_w.shape
        self.weight.data = pruned_w

        # TODO: hardcode now for conv2d and linear
        if pruned_dims is not None and pruned_dims[1] != None:
            self.input_prune_channels = prune_channels_multi_dims[1]
        return

    def extract_input(self, x, layer_type):
        # [batch_size, in_features] / [batch_size, seq_lens, token_lens]
        if layer_type == 'linear':
            if x.dim() != 2 and x.dim() != 3:
                raise ValueError('Not valid input dimension')
            
            # Create a boolean mask for all indices
            mask = torch.ones(x.size(-1), dtype=torch.bool)
            if self.input_prune_channels is None:
                return x
            # Mark the indices to be pruned as False
            mask[self.input_prune_channels] = False
            # Use the mask to index the tensor

            pruned_x = x.index_select(dim=-1, index=mask.nonzero().squeeze().to(x.device))
            return pruned_x
        elif layer_type == 'conv2d':
            if x.dim() != 4:
                raise ValueError('Not valid input dimension')
            
            # Create a boolean mask for all indices
            mask = torch.ones(x.size(1), dtype=torch.bool)
            if self.input_prune_channels is None:
                return x
            # Mark the indices to be pruned as False
            mask[self.input_prune_channels] = False
            # Use the mask to index the tensor

            pruned_x = x.index_select(dim=1, index=mask.nonzero().squeeze().to(x.device))
            return pruned_x
        else:
            raise ValueError('Not valid layer type')
    
    def extract_in_weight(self, input_dim, pruned_dims, prune_channels_multi_dims, layer_type, is_prune_out_dim=None):
        # unstruct pruning
        if pruned_dims is None:
            return self.weight
        
        if layer_type == 'linear':
            # [batch_size, in_features] / [batch_size, seq_lens, token_lens]
            if input_dim != 2 and input_dim != 3:
                raise ValueError('Not valid input dimension')
            
            if 'wandaunstrcut' in pruned_dims:
                # new_weight = self.weight.clone()

                src_tensor = torch.tensor(0, dtype=self.weight.dtype, device=self.weight.device)
                # Perform the scatter operation on the new tensor
                # new_weight.scatter_(dim=1, index=prune_channels_multi_dims, src=src_tensor)

                # Create a mask with all elements set to True
                mask = torch.ones_like(self.weight, dtype=self.weight.dtype, device=self.weight.device)

                # Set the specified indices in the mask to False
                mask.scatter_(dim=1, index=prune_channels_multi_dims, src=src_tensor)

                # Apply the mask to the tensor
                weight = self.weight * mask

                # Return the modified copy
                return weight
            
            if pruned_dims[-1] is None or prune_channels_multi_dims[-1] is None:
                # print('zzzzz')
                return self.weight
            else:
                extract_weight_dim = 1
                # weight dim 1 should have same size as original h dim 2
                mask = torch.ones(self.weight.size(extract_weight_dim), dtype=torch.bool)
                # Mark the indices to be pruned as False
                mask[prune_channels_multi_dims[-1]] = False
                # Use the mask to index the tensor
                weight = torch.index_select(self.weight, dim=extract_weight_dim, index=mask.nonzero().squeeze().to(self.weight.device))
                return weight
           
        
    def extract_out_weight(self, input_dim, pruned_dims, prune_channels_multi_dims, layer_type, is_prune_out_dim=None):
        # unstruct pruning
        if pruned_dims is None:
            return self.weight
        
        if layer_type == 'linear':
            # [batch_size, in_features] / [batch_size, seq_lens, token_lens]
            if input_dim != 2 and input_dim != 3:
                raise ValueError('Not valid input dimension')
            
            if 'wandaunstrcut' in pruned_dims:
                # new_weight = self.weight.clone()

                src_tensor = torch.tensor(0, dtype=self.weight.dtype, device=self.weight.device)
                # Perform the scatter operation on the new tensor
                # new_weight.scatter_(dim=1, index=prune_channels_multi_dims, src=src_tensor)

                # Create a mask with all elements set to True
                mask = torch.ones_like(self.weight, dtype=self.weight.dtype, device=self.weight.device)

                # Set the specified indices in the mask to False
                mask.scatter_(dim=1, index=prune_channels_multi_dims, src=src_tensor)

                # Apply the mask to the tensor
                weight = self.weight * mask

                # Return the modified copy
                return weight
            
            if pruned_dims[-1] is None or prune_channels_multi_dims[-1] is None:
                # print('zzzzz')
                return self.weight
            else:
                extract_weight_dim = 0
                # weight dim 1 should have same size as original h dim 2
                mask = torch.ones(self.weight.size(extract_weight_dim), dtype=torch.bool)
                # Mark the indices to be pruned as False
                mask[prune_channels_multi_dims[-1]] = False

                self.out_selected_dim = mask
                # print('self.out_selected_dim', self.out_selected_dim)
                # Use the mask to index the tensor
                weight = torch.index_select(self.weight, dim=extract_weight_dim, index=mask.nonzero().squeeze().to(self.weight.device))
                return weight
        
class Linear(nn.Linear, EriLayer):
    def __init__(
        self,
        prune_name,
        in_features,
        out_features,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        self.prune_name = prune_name
        nn.Linear.__init__(self, in_features, out_features, bias=False)
        EriLayer.__init__(self, in_features=in_features, out_features=out_features, **kwargs)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        # print('fan_in_fan_out: ', fan_in_fan_out, self.weight.data.shape, self.weight.shape)
        # GPT2 has CON1D, which is a self-defined layer, not the traditional Conv1D
        # print('self.weight')
        # if fan_in_fan_out:
        #     self.weight.data = self.weight.data.T
        # print('after fan_in_fan_out: ', fan_in_fan_out, self.weight.data.shape)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.layer_type = 'linear'
        
        self.prune_metric = cfg['prune_metric']
        # if 'local' in self.prune_name and self.prune_tgt == 'weight':
        #     self.prune_weight(self.weight, self.layer_type)
    

    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            previous_dtype = x.dtype
            if self.prune_tgt == 'hidden_repr':
                if 'fcst' in cfg['prune_name'] and 'cal_mlp_fcst_out_dim_metric' in kwargs:
                    # broadcast and return out_dim * in_dim matrix
                    return x * self.weight
                elif 'fcst' in cfg['prune_name'] and 'cal_attn_fcst_out_dim_metric' in kwargs:
                    # need to save s dimension for the following fcst
                    return F.linear(x, self.weight, bias=self.bias)
                # print('-----\n')
                # print('input_shape: ', x.shape)
                # print("prev weight.shape", self.weight.shape)
                batch_size = x.size(0)
                input_dim = x.dim()
                seq_len = x.size(1)
                input_shape = x.shape
                # print('input_shape: ', input_shape, self.key)
                linear_layer_info = {
                    'weight': self.weight.data,
                    # 'weight_norm_across_channel_dims': self.weight_norm_across_channel_dims,
                }
                # if 'fcst' in cfg['prune_name'] and self.is_prune_out_dim == False:
                # print('ceshi','fcst' in cfg['prune_name'], 'fcst_out_dim_indices' in kwargs )
                # enter down-proj after delete outputdim of gate-proj and up-proj
                if 'WO' in cfg['prune_metric'] and self.is_prune_out_dim == False:
                    # print('here1')
                    if 'fcst' in cfg['prune_name']:
                        # if it is forecast and parallel, we consider all the structure and operations
                        # we already know the weight index to extract
                        weight = torch.index_select(self.weight, dim=1, index=kwargs['fcst_out_dim_indices'].to(self.weight.device))
                    else:
                        # prune out dim situation
                        # but not forecast (only consider current layer's input and weight)
                        # or forecast, but do not consider parallel strucutre's effect, only considering gate or up individually.
                        non_zero_indices = torch.nonzero(torch.sum(x, dim=(0,1)), as_tuple=True)[-1]
                        weight = torch.index_select(self.weight, dim=1, index=non_zero_indices.to(self.weight.device))
                        x = x[..., non_zero_indices]
                # for forecast situation, entering up-proj or gate-proj
                elif 'fcst' in cfg['prune_name'] and 'fcst_out_dim_indices' in kwargs:
                    # # weight dim 1 should have same size as original h dim 2
                    # mask = torch.ones(self.weight.size(0), dtype=torch.bool)
                    # # Mark the indices to be pruned as False
                    # mask[kwargs['fcst_out_dim_indices']] = False
                    
                    # Use the mask to index the tensor
                    # print('here2')
                    weight = torch.index_select(self.weight, dim=0, index=kwargs['fcst_out_dim_indices'].to(self.weight.device))
                    # print('weight.shape', weight.shape)
                    # record to fill zero
                    if 'para' not in cfg['prune_name']:
                        self.out_selected_dim = kwargs['fcst_out_dim_indices']
                else:
                    # print('here3')
                    x, pruned_dims, prune_channels_multi_dims = self.pruning_module.batch_pruning(x, self.layer_type, linear_layer_info, self.key, self.is_prune_out_dim)
                    if 'WO' in cfg['prune_metric']:
                        weight = self.extract_out_weight(input_dim, pruned_dims, prune_channels_multi_dims, self.layer_type)
                    else:
                        weight = self.extract_in_weight(input_dim, pruned_dims, prune_channels_multi_dims, self.layer_type)
                
                result = F.linear(x, weight, bias=self.bias)
                # refill the pruned output dim with 0
                # gate-proj & up-proj
                if 'WO' in cfg['prune_metric'] and self.is_prune_out_dim == True:
                    # dont need to refill 0 because all indices are determined
                    if 'fcst' in cfg['prune_name']:
                        pass
                    else:
                        recovered_output = torch.zeros(batch_size, seq_len, self.out_features, device=result.device, dtype=result.dtype)
                        # Insert the output from the pruned layer into the correct positions in the extended tensor
                        # print('extended_output.shape', extended_output.shape)
                        # print('key', self.key)
                        # print('self.out_selected_dim', torch.nonzero(self.out_selected_dim).squeeze())
                        # print('gate&up', (self.out_selected_dim.shape[0] - torch.sum(self.out_selected_dim))/self.weight.shape[0], self.key)
                        recovered_output[..., self.out_selected_dim] = result

                        # 'extended_output' now has the pruned outputs filled in and zeros elsewhere
                        result = recovered_output
            # elif self.prune_tgt == 'weight':
            #     pruned_h = self.extract_input(x, self.layer_type)
            #     result = F.linear(pruned_h, self.weight, bias=self.bias)
            
            # self.pruning_module.cal_repr_distribution(pruned_h, f'{self.key}_pruned_hist')
            result = result.to(previous_dtype)
            # print('result.shape: ', result.shape, self.key)
        return result
    

class Conv2d(nn.Conv2d, EriLayer):
    def __init__(
        self,
        prune_name,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        **kwargs,
    ):
        self.prune_name = prune_name
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        EriLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            **kwargs
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False   

        self.layer_type = 'conv2d'
        # if 'local' in self.prune_name and self.prune_tgt == 'weight':
        #     self.prune_weight(self.weight, self.layer_type)
            
    def forward(self, x: torch.Tensor):
        # print('input_shape: ', x.shape)
        # print('pruned_conv2d')
        previous_dtype = x.dtype
        if self.prune_tgt == 'hidden_repr':
            # print('-----\n')
            # print('input_shape: ', x.shape)
            # print("prev weight.shape", self.weight.shape)
            input_dim = x.dim()
            conv2d_layer_info = {
                'weight': self.weight.data,
            }
            x, pruned_dims, prune_channels_multi_dims = self.pruning_module.batch_pruning(x, self.layer_type, conv2d_layer_info, self.key)
            weight = self.extract_weight(input_dim, pruned_dims, prune_channels_multi_dims, self.layer_type)

            result = F.conv2d(
                x,
                weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.prune_tgt == 'weight':
            x = self.extract_input(x, self.layer_type)
            # b = self.weight.shape
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        # self.pruning_module.cal_repr_distribution(pruned_h, f'{self.key}_pruned_hist')
        result = result.to(previous_dtype)
        return result



# def linear_flops_compute(self, input, weight, bias=None):
#         out_features = weight.shape[0]
#         macs = input.numel() * out_features
#         return 2 * macs, macs

#     def cal_saving_flops(self, h, prune_dim, prune_channels, layer_type, layer_info):
#         if prune_channels is None:
#             return 0
#         if layer_type == 'linear':
#             weight = layer_info['weight']

#             rest_dim_sizes = [h.shape[i] for i in range(h.dim()) if i != prune_dim]
#             product_of_rest_dims = prod(rest_dim_sizes)

#             out_features = weight.shape[0]
#             saving_flops = 2 * len(prune_channels) * product_of_rest_dims * out_features
#         elif layer_type == 'conv2d':
#             saving_flops = 0
#         else:
#             raise ValueError('Not valid layer type')
#         # rest_dims = tuple(i for i in range(h.dim()) if i != prune_dim)
#         # prune_eles = len(prune_channels) * reduce(lambda x, y: x * y, h.shape[rest_dims])
#         return saving_flops


    # def cal_saving_flops(self, h, prune_dim, prune_channels, layer_type, layer_info):
    #     if prune_channels is None:
    #         return 0
    #     if layer_type == 'linear':
    #         weight = layer_info['weight']

    #         rest_dim_sizes = [h.shape[i] for i in range(h.dim()) if i != prune_dim]
    #         product_of_rest_dims = prod(rest_dim_sizes)

    #         out_features = weight.shape[0]
    #         saving_flops = 2 * len(prune_channels) * product_of_rest_dims * out_features
    #     elif layer_type == 'conv2d':
    #         pass

    #     else:
    #         raise ValueError('Not valid layer type')
    #     # rest_dims = tuple(i for i in range(h.dim()) if i != prune_dim)
    #     # prune_eles = len(prune_channels) * reduce(lambda x, y: x * y, h.shape[rest_dims])
    #     return saving_flops




    #     # placeholder
    # def _conv_flops_compute(input_shape, weight_shape, bias=None, stride=1, padding=0, dilation=1, groups=1):
    #     assert weight.shape[1] * groups == input.shape[1]

    #     batch_size = input.shape[0]
    #     in_channels = input.shape[1]
    #     out_channels = weight.shape[0]
    #     kernel_dims = list(weight.shape[2:])
    #     input_dims = list(input.shape[2:])

    #     length = len(input_dims)

    #     strides = stride if type(stride) is tuple else (stride, ) * length
    #     dilations = dilation if type(dilation) is tuple else (dilation, ) * length
    #     if isinstance(padding, str):
    #         if padding == 'valid':
    #             paddings = (0, ) * length
    #         elif padding == 'same':
    #             paddings = ()
    #             for d, k in zip(dilations, kernel_dims):
    #                 total_padding = d * (k - 1)
    #                 paddings += (total_padding // 2, )
    #     elif isinstance(padding, tuple):
    #         paddings = padding
    #     else:
    #         paddings = (padding, ) * length

    #     output_dims = []
    #     for idx, input_dim in enumerate(input_dims):
    #         output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] *
    #                                                     (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
    #         output_dims.append(output_dim)

    #     filters_per_channel = out_channels // groups
    #     conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    #     active_elements_count = batch_size * int(_prod(output_dims))
    #     overall_conv_macs = conv_per_position_macs * active_elements_count
    #     overall_conv_flops = 2 * overall_conv_macs

    #     bias_flops = 0
    #     if bias is not None:
    #         bias_flops = out_channels * active_elements_count

    #     return int(overall_conv_flops + bias_flops), int(overall_conv_macs)