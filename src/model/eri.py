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
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from transformers.pytorch_utils import Conv1D
from config import cfg
from math import prod
from .model import init_param, mse_loss
from typing import List, Optional, Tuple, Union
from module import to_device, TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING
from functools import reduce

class EriModel(torch.nn.Module):
    def __init__(self, model, logger):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.add_pruner('pruner', logger)

    def add_pruner(self, pruner_name, logger):
        if cfg['prune_tgt'] == 'hidden_repr':
            self.pruning_module = HiddenRepresentationPruning(cfg, logger)
        elif cfg['prune_tgt'] == 'weight':
            self.pruning_module = WeightPruning(cfg, logger)
        self._find_and_replace(pruner_name)
        mark_no_trainable(self.model)
        return
    
    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        
    def _check_target_module_exists(self, target_modules, key):
        if isinstance(target_modules, str):
            target_module_found = re.fullmatch(target_modules, key)
        else:
            # target_module_found = any(key.endswith(target_key) for target_key in target_modules)
            target_module_found = any(key.endswith(target_key) for target_key in target_modules)

            # for target_key in target_modules:
            #     if key.endswith(target_key):
            #         target_module_found = True
            #         break

        return target_module_found

    def _create_new_module(self, pruner_name, target, key):
        bias = hasattr(target, "bias") and target.bias is not None
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        kwargs = {
            "prune_tgt": cfg['prune_tgt'],
            "pruning_module": self.pruning_module,
            "key": key,
            "fan_in_fan_out": False
        }
        # if isinstance(target, torch.nn.Embedding):
        #     embedding_kwargs = kwargs.copy()
        #     embedding_kwargs.pop("fan_in_fan_out", None)
        #     in_features, out_features = target.num_embeddings, target.embedding_dim
        #     new_module = Embedding(pruner_name, in_features, out_features, **embedding_kwargs)
        if isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            dilation = target.dilation
            groups = target.groups
            print(key, out_channels, in_channels, kernel_size, stride, padding, flush=True)
            new_module = Conv2d(pruner_name, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, **kwargs)
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
            new_module = Linear(pruner_name, in_features, out_features, bias=bias, **kwargs)

        return new_module

    def _find_and_replace(self, pruner_name):
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        target_modules = TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING[cfg['model_type']]
        if 'cust_tgt_modules' in cfg and cfg['cust_tgt_modules'] is not None:
            target_modules = cfg['cust_tgt_modules']
        print('target_modules: ', target_modules)
        for key in key_list:
            if not self._check_target_module_exists(target_modules, key):
                continue

            # TODO: hardcode
            if cfg['model_type'] == 'roberta':
                if cfg['cust_tgt_modules'] == ['output.dense'] and 'attention.output.dense' in key:
                    continue

            print('Replaced Layer Keys', key, flush=True)
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            
            new_module = self._create_new_module(pruner_name, target, key)
            self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING[cfg['model_type']]} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            # if old_module.bias is not None:
            new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "pruner_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def mark_no_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = False
    return

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


class EriLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.pruning_channel_ratio = []
        pass
    
class Linear(nn.Linear, EriLayer):
    def __init__(
        self,
        pruner_name,
        in_features,
        out_features,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        EriLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        # print('fan_in_fan_out: ', fan_in_fan_out, self.weight.data.shape, self.weight.shape)
        # GPT2 has CON1D, which is a self-defined layer, not the traditional Conv1D
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        # print('after fan_in_fan_out: ', fan_in_fan_out, self.weight.data.shape)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        
        self.prune_tgt = kwargs['prune_tgt']
        if self.prune_tgt == 'weight':
            self.prune_weight(self.weight)
        self.pruning_module = kwargs['pruning_module']
        self.key = kwargs['key']
        self.is_pruned = True
        

    # TODO, weight.data
    def prune_weight(self, weight):
        linear_layer_info = {
            'weight': self.weight,
            'bias': self.bias,
        }
        pruned_w, pruned_dims, prune_channels_multi_dims = self.pruning_module.batch_pruning(weight, 'linear', linear_layer_info)
        # print('pruned_w.shape: ', pruned_w.shape)
        self.weight = nn.Parameter(pruned_w)

        self.input_prune_channels = None
        if pruned_dims[1] != None:
            self.input_prune_channels = prune_channels_multi_dims[1]
        return 

    def extract_input(self, x):
        # [batch_size, in_features] / [batch_size, seq_lens, token_lens]
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
    
    def extract_weight(self, input_dim, pruned_h, pruned_dims, prune_channels_multi_dims):
        # unstruct pruning
        if pruned_dims is None:
            return self.weight
        
        # [batch_size, in_features] / [batch_size, seq_lens, token_lens]
        if input_dim != 2 and input_dim != 3:
            raise ValueError('Not valid input dimension')
        
        if pruned_dims[-1] is None or prune_channels_multi_dims[-1] is None:
            # print('zzzzz')
            return self.weight
        else:
            extract_weight_dim = 0 if self.fan_in_fan_out else 1
            # weight dim 1 should have same size as original h dim 2
            mask = torch.ones(self.weight.size(extract_weight_dim), dtype=torch.bool)
            # Mark the indices to be pruned as False
            mask[prune_channels_multi_dims[-1]] = False
            # Use the mask to index the tensor
            weight = torch.index_select(self.weight, dim=extract_weight_dim, index=mask.nonzero().squeeze().to(self.weight.device))
            return weight

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        print('pruned_linear')
        if self.prune_tgt == 'hidden_repr':
            # print('-----\n')
            # print('input_shape: ', x.shape)
            # print("prev weight.shape", self.weight.shape)
            input_dim = x.dim()
            input_shape = x.shape
            linear_layer_info = {
                'weight': self.weight,
                'bias': self.bias,
            }
            pruned_h, pruned_dims, prune_channels_multi_dims = self.pruning_module.batch_pruning(x, 'linear', linear_layer_info, self.key)
            weight = self.extract_weight(input_dim, pruned_h, pruned_dims, prune_channels_multi_dims)
            
            
            # print('pruned_dims: ', pruned_dims)
            # print("pruned_h.shape", pruned_h.shape)
            # print("pruned_dims", pruned_dims)
            # print("prune_channels_multi_dims", prune_channels_multi_dims)
            # print("weight.shape", weight.shape)
            result = F.linear(pruned_h, transpose(weight, self.fan_in_fan_out), bias=self.bias)
            # result = F.linear(pruned_h, weight, bias=self.bias)
        elif self.prune_tgt == 'weight':
            pruned_h = self.extract_input(x)
            result = F.linear(pruned_h, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        result = result.to(previous_dtype)
        return result
    

class Conv2d(nn.Conv2d, EriLayer):
    def __init__(
        self,
        pruner_name,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        EriLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False   

        self.prune_tgt = kwargs['prune_tgt']
        if self.prune_tgt == 'weight':
            self.prune_weight(self.weight)
        self.pruning_module = kwargs['pruning_module']
        self.key = kwargs['key']
        
        # self.hook = self.register_forward_hook(self.forward_hook)
    
    # def forward_hook(self, module, input, output):
    #     print('key', module.key)
    #     print('input', input[0][0][0])
    #     print('output', output[0][0][0])
    #     print('weight', module.weight[0][0][0], module.weight[1][0][0], module.stride, module.padding, module.dilation, module.groups, flush=True)
    #     print('bias', module.bias, flush=True)
    #     return
    
    def prune_weight(self, weight):
        conv2d_layer_info = {
            'weight': self.weight,
            'bias': self.bias,
        }
        pruned_w, pruned_dims, prune_channels_multi_dims = self.pruning_module.batch_pruning(weight, 'conv2d', conv2d_layer_info)
        # print('pruned_w.shape: ', pruned_w.shape)
        self.weight = nn.Parameter(pruned_w)

        self.input_prune_channels = None
        if pruned_dims[1] != None:
            self.input_prune_channels = prune_channels_multi_dims[1]
        return 

    def extract_input(self, x):
        # [batch_size, in_channels, h, w]
        if x.dim() != 4:
            raise ValueError('Not valid input dimension')
        
        # Create a boolean mask for all indices
        mask = torch.ones(x.size(-1), dtype=torch.bool)
        if self.input_prune_channels is None:
            return x
        # Mark the indices to be pruned as False
        mask[self.input_prune_channels] = False
        # Use the mask to index the tensor

        pruned_x = x.index_select(dim=1, index=mask.nonzero().squeeze().to(x.device))
        return pruned_x
    
    def extract_weight(self, input_dim, pruned_h, pruned_dims, prune_channels_multi_dims):
        # unstruct pruning
        if pruned_dims is None:
            return self.weight
        
         # [batch_size, in_channels, h, w]
        if input_dim != 4:
            raise ValueError('Not valid input dimension')
        
        if pruned_dims[1] is None or prune_channels_multi_dims[1] is None:
            # print('zzzzz')
            return self.weight
        else:
            extract_weight_dim = 1
            # weight dim 1 should have same size as original h dim 2
            mask = torch.ones(self.weight.size(extract_weight_dim), dtype=torch.bool)
            # Mark the indices to be pruned as False
            mask[prune_channels_multi_dims[1]] = False
            # Use the mask to index the tensor
            weight = torch.index_select(self.weight, dim=extract_weight_dim, index=mask.nonzero().squeeze().to(self.weight.device))
            return weight
        
    def forward(self, x: torch.Tensor):
        print('pruned_conv2d')
        previous_dtype = x.dtype
        if self.prune_tgt == 'hidden_repr':
            # print('-----\n')
            # print('input_shape: ', x.shape)
            # print("prev weight.shape", self.weight.shape)
            input_dim = x.dim()
            conv2d_layer_info = {
                'weight': self.weight,
                'bias': self.bias,
            }
            # pruned_h, pruned_dims, prune_channels_multi_dims = self.pruning_module.batch_pruning(x, 'conv2d', conv2d_layer_info, self.key)
            # weight = self.extract_weight(input_dim, pruned_h, pruned_dims, prune_channels_multi_dims)
            # input_is_equal = torch.equal(x, pruned_h)
            # weight_is_equal = torch.equal(self.weight, weight)

            # pruned_h = x
            # weight = self.weight
            # result = F.conv2d(
            #     pruned_h,
            #     weight,
            #     bias=self.bias,
            #     stride=self.stride,
            #     padding=self.padding,
            #     dilation=self.dilation,
            #     groups=self.groups,
            # )
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            # result = F.linear(pruned_h, weight, bias=self.bias)
        elif self.prune_tgt == 'weight':
            # pruned_h = self.extract_input(x)
            # result = F.conv2d(
            #     pruned_h,
            #     self.weight,
            #     bias=self.bias,
            #     stride=self.stride,
            #     padding=self.padding,
            #     dilation=self.dilation,
            #     groups=self.groups,
            # )
            pass
        result = result.to(previous_dtype)
        return result

class BasePruning:
    def __init__(self, cfg, logger):
        self.prune_name = cfg['prune_name']
        self.prune_tgt = cfg['prune_tgt']
        self.prune_norm = cfg['prune_norm']
        self.prune_hyper = cfg['prune_hyper'] 
        self.prune_dim = cfg['prune_dim'] 
        self.prune_dim_select_mode = cfg['prune_dim_select_mode'] 
        self.batch_integ = cfg['batch_integ']
        self.logger = logger


class HiddenRepresentationPruning(BasePruning):

    def __init__(self, cfg, logger):
        BasePruning.__init__(self, cfg, logger)
        pass
        
    def batch_pruning(self, h, layer_type, layer_info, key):
        # Exclude the first dimension (batch size) and the prune_dim
        # calculate the pq-index per sample
        exclude_dim = 0 if self.batch_integ in ['inter', 'union'] else None
        prune_channels_multi_dims = [None] * h.dim()
        saving_flops_multi_dims = [0] * h.dim()
        h_shape = h.shape
        h_type = h.dtype
        
        if 'unstruct' in self.prune_name:
            if self.batch_integ not in ['inter', 'union']:
                raise ValueError('Not valid batch integration method')
            if self.prune_name == 'base-unstruct':
                prune_indices = self.apply_pruning(h)
                prune_indices = self.apply_batch_integ(prod(h_shape[1:]), prune_indices)
                # Create a mask with the same shape as h, initially set to True
                mask = torch.ones_like(h, dtype=torch.bool)
                if prune_indices is not None:
                    # Mark the indices to be pruned as False
                    mask[:, prune_indices] = False
                pruned_h = h * mask.to(h.device)
                return pruned_h, None, None
            elif self.prune_name == 'pq-unstruct':
                pass
        elif 'struct' in self.prune_name:
            if self.prune_dim_select_mode == 'max':
                for prune_dim in self.prune_dim:
                    prune_channels = self.apply_pruning(h, key, prune_dim, exclude_dim)
                    prune_channels = self.apply_batch_integ(h_shape[prune_dim], prune_channels)
                    prune_channels_multi_dims[prune_dim] = prune_channels

                    saving_flops = self.cal_saving_flops(h, prune_dim, prune_channels, layer_type, layer_info)
                    saving_flops_multi_dims[prune_dim] = saving_flops

                if len(self.prune_dim) == 1:
                    final_prune_dim = self.prune_dim[0]
                else:
                    raise ValueError('Not valid prune dim')
                final_prune_dim = np.argmax(saving_flops_multi_dims)
                # print('saving_flops_multi_dims', saving_flops_multi_dims)
                # print('prune_channels_multi_dims', prune_channels_multi_dims)
                # final_prune_dim = 2
                pruned_h = self.prune_h(h, final_prune_dim, prune_channels_multi_dims[final_prune_dim])

                # print('final_prune_dim', final_prune_dim)
                # print('saving_flops_multi_dims', saving_flops_multi_dims)
                # print('prune_channels_multi_dims', prune_channels_multi_dims)

                pruned_dims = [dim if dim == final_prune_dim else None for dim in range(h.dim())]
                for dim in range(len(prune_channels_multi_dims)):
                    if dim != final_prune_dim:
                        prune_channels_multi_dims[dim] = None
                        saving_flops_multi_dims[dim] = 0
                
                # TODO: hardcode
                if prune_channels_multi_dims[pruned_dims[final_prune_dim]] == None:
                    num_pruned_channels = 0
                else:
                    num_pruned_channels = prune_channels_multi_dims[pruned_dims[final_prune_dim]].size(-1)

                cur_batch_info = {
                    f"{key}_pruned_dims": pruned_dims[final_prune_dim],
                    f"{key}_pruned_channels": num_pruned_channels,
                    f"{key}_total_channels": h_shape[pruned_dims[final_prune_dim]],
                    f"{key}_pruned_ratio": num_pruned_channels / (h_shape[pruned_dims[final_prune_dim]] + 1e-10),
                }
                self.logger.append(cur_batch_info, 'test', 1)
                # print('after\n')
                # print('final_prune_dim', final_prune_dim)
                # print('saving_flops_multi_dims', saving_flops_multi_dims)
                # print('prune_channels_multi_dims', prune_channels_multi_dims)
                return pruned_h, pruned_dims, prune_channels_multi_dims
            elif self.prune_dim_select_mode == 'casc':
                for prune_dim in self.prune_dim:
                    prune_channels = self.apply_pruning(h, prune_dim, exclude_dim)
                    prune_channels = self.apply_batch_integ(h_shape[prune_dim], prune_channels)
                    prune_channels_multi_dims[prune_dim] = prune_channels
                    
                    saving_flops = self.cal_saving_flops(h, prune_dim, prune_channels, layer_type, layer_info)
                    saving_flops_multi_dims[prune_dim] = saving_flops

                    pruned_h = self.prune_h(h, prune_dim, prune_channels_multi_dims[prune_dim])
                    h = pruned_h
                pruned_dims = [dim if dim in self.prune_dim else None for dim in range(h.dim())]
                return pruned_h, pruned_dims, prune_channels_multi_dims
        else:
            raise ValueError('Not valid pruning method')
        
    def prune_h(self, h, prune_dim, prune_channels):
        # Create a boolean mask for all indices
        mask = torch.ones(h.size(prune_dim), dtype=torch.bool)
        # print('pre_mask', mask)
        # Mark the indices to be pruned as False
        if prune_channels is not None:
            mask[prune_channels] = False
        # print('mask', mask, prune_channels)
        # Use the mask to index the tensor
        pruned_h = h.index_select(dim=prune_dim, index=mask.nonzero().squeeze().to(h.device))
        return pruned_h

    def linear_flops_compute(self, input, weight, bias=None):
        out_features = weight.shape[0]
        macs = input.numel() * out_features
        return 2 * macs, macs

    def cal_saving_flops(self, h, prune_dim, prune_channels, layer_type, layer_info):
        if prune_channels is None:
            return 0
        if layer_type == 'linear':
            weight = layer_info['weight']

            rest_dim_sizes = [h.shape[i] for i in range(h.dim()) if i != prune_dim]
            product_of_rest_dims = prod(rest_dim_sizes)

            out_features = weight.shape[0]
            saving_flops = 2 * len(prune_channels) * product_of_rest_dims * out_features
        elif layer_type == 'conv2d':
            saving_flops = 0
        else:
            raise ValueError('Not valid layer type')
        # rest_dims = tuple(i for i in range(h.dim()) if i != prune_dim)
        # prune_eles = len(prune_channels) * reduce(lambda x, y: x * y, h.shape[rest_dims])
        return saving_flops
    
    def apply_batch_integ(self, cur_total_channels, prune_channels):
        if prune_channels[0].numel() == 0:  # Check if the tensor is empty
            return None

        if self.batch_integ == 'inter':
            sets = [set(tensor.tolist()) for tensor in prune_channels]
            if len(sets) == 0:
                sets = [set()]
            intersected_set = set.intersection(*sets)
            prune_channels = torch.tensor(list(intersected_set), dtype=torch.long)
        elif self.batch_integ == 'union':
            sets = [set(tensor.tolist()) for tensor in prune_channels]
            if len(sets) == 0:
                sets = [set()]
            intersected_set = set.union(*sets)
            prune_channels = torch.tensor(list(intersected_set), dtype=torch.long)
        elif self.batch_integ == 'full':
            prune_channels = torch.tensor(prune_channels[0], dtype=torch.long)
        else:
            raise ValueError('Not valid batch integration method')
        if prune_channels.numel() == 0:
            return None
        if prune_channels.numel() >= cur_total_channels:
            prune_channels_list = prune_channels.tolist()
            prune_channels_list.remove(random.choice(prune_channels_list))
            # Convert back to tensor
            prune_channels = torch.tensor(prune_channels_list, dtype=torch.long)
            warnings.warn("Attempting to prune all channels. Keeping one channel for calculation.")
        return prune_channels
    
    def apply_pruning(self, h, key, prune_dim=None, exclude_dim=None):
        if prune_dim >= h.dim():
            raise ValueError('Not valid pruning dimension')
        if exclude_dim is not None and exclude_dim != 0:
            raise ValueError('Not valid exclude dimension')
        # No pruning
        if self.prune_hyper == 9999:
            return [torch.empty(0)]
        
        if self.prune_name == 'pqstruct':
            prune_channels = self.pq_struct(h, key, prune_dim, exclude_dim)
            return prune_channels
        elif self.prune_name == 'basestruct':
            prune_channels = self.base_struct(h, key, prune_dim, exclude_dim)
            return prune_channels
        elif self.prune_name == 'baseunstruct':
            pruned_indices = self.base_unstruct(h, key)
            return pruned_indices
        else:
            raise ValueError('Not valid pruning method')
        

    def pq_struct(self, h, key, prune_dim, exclude_dim):
        # set pq-index hyper
        p = 1
        q = 2
        gamma = 1
        beta = 0.9
        eta = self.prune_hyper
        calc_dim = 1 if exclude_dim == 0 else 0
        
        dims_to_aggregate = tuple(i for i in range(h.dim()) if i != prune_dim and i != exclude_dim)
        norm_across_other_dims = torch.linalg.vector_norm(h, ord=self.prune_norm, dim=dims_to_aggregate)     

        norm_p = torch.linalg.vector_norm(norm_across_other_dims, ord=p, dim=calc_dim)
        norm_q = torch.linalg.vector_norm(norm_across_other_dims, ord=q, dim=calc_dim) + 1e-10
        
        dimension = h.shape[prune_dim]
        pq_indices = (1 - dimension ** (1/q - 1/p) * norm_p / norm_q)

        # add additional dimension if dimension is 0
        if pq_indices.dim() == 0:
            pq_indices = pq_indices.unsqueeze(0)

        if torch.isnan(pq_indices).any():
            raise ValueError('pq_indices contains nan values')

        lower_bound = dimension * (1 + eta) ** (-q / (q - p)) * (1 - pq_indices) ** (q * p / (q - p))
        beta_tensor = torch.full_like(lower_bound, beta)
        prune_channels_count = torch.floor(dimension * torch.min(gamma * (1 - lower_bound / dimension), beta_tensor))

        _, sorted_channels = torch.sort(norm_across_other_dims, dim=calc_dim)
        prune_channels = [sorted_channels[i, :int(count.item())] for i, count in enumerate(prune_channels_count)]
        info = {
            f"{key}_norm_across_other_dims": norm_across_other_dims.mean(dim=0).squeeze(0).tolist(),
            f"{key}_pq_indices": pq_indices.mean(dim=0).squeeze(0).tolist(),
        }
        self.logger.append(info, 'test', h.shape[0])   
        return prune_channels

    def base_unstruct(self, h):
        flattened_h = h.view(h.size(0), -1)
        norm_along_dim_1 = torch.linalg.vector_norm(flattened_h, ord=self.prune_norm, dim=1)
        _, sorted_indices = torch.sort(norm_along_dim_1, dim=1)
        num_indices_to_prune = int(self.prune_hyper * sorted_indices.size(1))
        # Select the indices to prune (lowest norms)
        pruned_indices = sorted_indices[:, :num_indices_to_prune]
        return pruned_indices
    
    def base_struct(self, h, prune_dim, exclude_dim):
        calc_dim = 1 if exclude_dim == 0 else 0
        dims_to_aggregate = tuple(i for i in range(h.dim()) if i != prune_dim and i != exclude_dim)
        norm_across_other_dims = torch.linalg.vector_norm(h, ord=self.prune_norm, dim=dims_to_aggregate)        
        prune_channels_count = int(self.prune_hyper * h.shape(prune_dim))
        _, sorted_channels = torch.sort(norm_across_other_dims, dim=calc_dim)
        prune_channels = [sorted_channels[i, :int(count.item())] for i, count in enumerate(prune_channels_count)]
        return prune_channels

        
    # placeholder
    def _conv_flops_compute(input_shape, weight_shape, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert weight.shape[1] * groups == input.shape[1]

        batch_size = input.shape[0]
        in_channels = input.shape[1]
        out_channels = weight.shape[0]
        kernel_dims = list(weight.shape[2:])
        input_dims = list(input.shape[2:])

        length = len(input_dims)

        strides = stride if type(stride) is tuple else (stride, ) * length
        dilations = dilation if type(dilation) is tuple else (dilation, ) * length
        if isinstance(padding, str):
            if padding == 'valid':
                paddings = (0, ) * length
            elif padding == 'same':
                paddings = ()
                for d, k in zip(dilations, kernel_dims):
                    total_padding = d * (k - 1)
                    paddings += (total_padding // 2, )
        elif isinstance(padding, tuple):
            paddings = padding
        else:
            paddings = (padding, ) * length

        output_dims = []
        for idx, input_dim in enumerate(input_dims):
            output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] *
                                                        (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
            output_dims.append(output_dim)

        filters_per_channel = out_channels // groups
        conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
        active_elements_count = batch_size * int(_prod(output_dims))
        overall_conv_macs = conv_per_position_macs * active_elements_count
        overall_conv_flops = 2 * overall_conv_macs

        bias_flops = 0
        if bias is not None:
            bias_flops = out_channels * active_elements_count

        return int(overall_conv_flops + bias_flops), int(overall_conv_macs)
















class WeightPruning(BasePruning):

    def __init__(self, cfg):
        BasePruning.__init__(self, cfg)
        pass

    def batch_pruning(self, w, layer_type, layer_info):
        if self.batch_integ != 'full':
            raise ValueError('Not valid batch integration method')
        if layer_type == 'linear' and self.prune_dim != [1]:
            raise ValueError('Not valid pruning dimension')

        prune_channels_multi_dims = [None] * w.dim()
        saving_flops_multi_dims = [0] * w.dim()
        w_shape = w.shape
        w_type = w.dtype
        prune_dim = 1
        final_prune_dim = 1

        if self.prune_name == 'base-unstruct':
            prune_indices = self.apply_pruning(w)
            prune_indices = self.apply_batch_integ(prod(w_shape), prune_indices)
            # Create a mask with the same shape as h, initially set to True
            mask = torch.ones_like(w, dtype=torch.bool)
            if prune_indices is not None:
                mask[:, prune_indices] = False
            pruned_w = w * mask.to(h.device)
            return pruned_w, None, None
        elif self.prune_name in ['pq', 'base-struct']:
            prune_channels = self.apply_pruning(w, prune_dim)
            prune_channels = self.apply_batch_integ(w_shape[prune_dim], prune_channels)
            prune_channels_multi_dims[prune_dim] = prune_channels

            pruned_w = self.prune_w(w, prune_dim, prune_channels_multi_dims[prune_dim])

            pruned_dims = [dim if dim == final_prune_dim else None for dim in range(w.dim())]
            for dim in range(len(prune_channels_multi_dims)):
                if dim != final_prune_dim:
                    prune_channels_multi_dims[dim] = None
                    # saving_flops_multi_dims[dim] = 0
            
            return pruned_w, pruned_dims, prune_channels_multi_dims
    
    def prune_w(self, w, prune_dim, prune_channels, mask=None):
        # Create a boolean mask for all indices
        mask = torch.ones(w.size(prune_dim), dtype=torch.bool)
        # Mark the indices to be pruned as False
        if prune_channels is not None:
            mask[prune_channels] = False
        # Use the mask to index the tensor
        pruned_w = w.index_select(dim=prune_dim, index=mask.nonzero().squeeze().to(h.device))
        return pruned_w

    def cal_saving_flops(self, h, prune_dim, prune_channels, layer_type, layer_info):
        if prune_channels is None:
            return 0
        if layer_type == 'linear':
            weight = layer_info['weight']

            rest_dim_sizes = [h.shape[i] for i in range(h.dim()) if i != prune_dim]
            product_of_rest_dims = prod(rest_dim_sizes)

            out_features = weight.shape[0]
            saving_flops = 2 * len(prune_channels) * product_of_rest_dims * out_features
        elif layer_type == 'conv2d':
            pass

        else:
            raise ValueError('Not valid layer type')
        # rest_dims = tuple(i for i in range(h.dim()) if i != prune_dim)
        # prune_eles = len(prune_channels) * reduce(lambda x, y: x * y, h.shape[rest_dims])
        return saving_flops
    
    def apply_batch_integ(self, cur_total_channels, prune_channels):
        if not prune_channels[0].numel():  # Check if the tensor is empty
            return None
        
        if self.batch_integ == 'full':
            prune_channels = torch.tensor(prune_channels[0], dtype=torch.long)
        else:
            raise ValueError('Not valid batch integration method')
        if prune_channels.numel() == 0:
            return None
        if prune_channels.numel() >= cur_total_channels:
            prune_channels_list = prune_channels.tolist()
            prune_channels_list.remove(random.choice(prune_channels_list))
            # Convert back to tensor
            prune_channels = torch.tensor(prune_channels_list, dtype=torch.long)
            warnings.warn("Attempting to prune all channels. Keeping one channel for calculation.")
        return prune_channels
    
    def apply_pruning(self, h, prune_dim=None):
        if prune_dim >= h.dim():
            raise ValueError('Not valid pruning dimension')
        
        if self.prune_name == 'pq':
            prune_channels = self.pq(h, prune_dim)
            return prune_channels
        elif self.prune_name == 'base-struct':
            prune_channels = self.base_struct(h, prune_dim)
            return prune_channels
        elif self.prune_name == 'base-unstruct':
            pruned_indices = self.base_unstruct(h)
            return pruned_indices
        else:
            raise ValueError('Not valid pruning method')
        

    def pq(self, w, prune_dim):
        # set pq-index hyper
        p = 1
        q = 2
        gamma = 1
        beta = 0.9
        eta = self.prune_hyper

        dims_to_aggregate = tuple(i for i in range(w.dim()) if i != prune_dim)
        norm_across_other_dims = torch.linalg.vector_norm(w, ord=self.prune_norm, dim=dims_to_aggregate)        
        norm_p = torch.linalg.vector_norm(norm_across_other_dims, ord=p, dim=0)
        norm_q = torch.linalg.vector_norm(norm_across_other_dims, ord=q, dim=0) + 1e-10
        
        dimension = w.shape[prune_dim]
        pq_indices = (1 - dimension ** (1/q - 1/p) * norm_p / norm_q)

        # add additional dimension if dimension is 0
        if pq_indices.dim() == 0:
            pq_indices = pq_indices.unsqueeze(0)

        if torch.isnan(pq_indices).any():
            raise ValueError('pq_indices contains nan values')

        lower_bound = dimension * (1 + eta) ** (-q / (q - p)) * (1 - pq_indices) ** (q * p / (q - p))
        beta_tensor = torch.full_like(lower_bound, beta)
        prune_channels_count = torch.floor(dimension * torch.min(gamma * (1 - lower_bound / dimension), beta_tensor))

        _, sorted_channels = torch.sort(norm_across_other_dims, dim=0)
        prune_channels = [sorted_channels[:int(prune_channels_count.item())]]
        return prune_channels

    def base_unstruct(self, w):
        flattened_w = w.view(-1)
        norm = torch.linalg.vector_norm(flattened_w, ord=self.prune_norm, dim=0)
        _, sorted_indices = torch.sort(norm, dim=0)
        num_indices_to_prune = int(self.prune_hyper * sorted_indices.size(0))
        # Select the indices to prune (lowest norms)
        pruned_indices = sorted_indices[:, :num_indices_to_prune]
        return pruned_indices
    
    def base_struct(self, w, prune_dim):
        dims_to_aggregate = tuple(i for i in range(w.dim()) if i != prune_dim)
        norm_across_other_dims = torch.linalg.vector_norm(w, ord=self.prune_norm, dim=dims_to_aggregate)        
        prune_channels_count = int(self.prune_hyper * w.shape(prune_dim))
        _, sorted_channels = torch.sort(norm_across_other_dims, dim=0)
        prune_channels = [sorted_channels[:int(prune_channels_count.item())]]
        return prune_channels


