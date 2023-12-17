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
from module import to_device, TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING
from functools import reduce

class EriModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.add_pruner('pruner')

    def add_pruner(self, pruner_name):
        self._find_and_replace(pruner_name)
        mark_no_trainable(self.model)
        if 'global' in cfg['prune_name'] and cfg['prune_tgt'] == 'weight':
            pruning_module = WeightPruning(cfg, 'global')
            pruning_module.global_pruning(self.model)
        return
    
    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

    def _create_new_module(self, pruner_name, target, key):
        bias = hasattr(target, "bias") and target.bias is not None
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if cfg['prune_tgt'] == 'hidden_repr':
            pruning_module = HiddenRepresentationPruning(cfg, key)
        elif cfg['prune_tgt'] == 'weight':
            pruning_module = WeightPruning(cfg, key)
        else:
            raise ValueError('Not valid prune tgt')
        
        kwargs = {
            "prune_tgt": cfg['prune_tgt'],
            "prune_name": cfg['prune_name'],
            "pruning_module": pruning_module,
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

        if 'local' in cfg['prune_name'] and cfg['prune_tgt'] == 'weight':
            new_module.prune_weight(new_module.weight, new_module.layer_type)

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

def _get_target_modules(cfg):
    target_modules = TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING[cfg['model_type']]
    if 'cust_tgt_modules' in cfg and cfg['cust_tgt_modules'] is not None:
        target_modules = cfg['cust_tgt_modules']
    return target_modules

global_only_one_module = True
def _check_target_module_exists(target_modules, key):
    if isinstance(target_modules, str):
        target_module_found = re.fullmatch(target_modules, key)
    else:
        # target_module_found = any(key.endswith(target_key) for target_key in target_modules)
        target_module_found = any(key.endswith(target_key) for target_key in target_modules)

    # TODO: hardcode for roberta
    if cfg['model_type'] == 'roberta':
        if cfg['cust_tgt_modules'] == ['output.dense'] and 'attention.output.dense' in key:
            return False

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
        self.pruning_channel_ratio = []
        self.input_prune_channels = None
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
    
    def extract_weight(self, input_dim, pruned_dims, prune_channels_multi_dims, layer_type):
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
                extract_weight_dim = 0 if self.fan_in_fan_out else 1
                # weight dim 1 should have same size as original h dim 2
                mask = torch.ones(self.weight.size(extract_weight_dim), dtype=torch.bool)
                # Mark the indices to be pruned as False
                mask[prune_channels_multi_dims[-1]] = False
                # Use the mask to index the tensor
                weight = torch.index_select(self.weight, dim=extract_weight_dim, index=mask.nonzero().squeeze().to(self.weight.device))
                return weight
        elif layer_type == 'conv2d':
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
                # print('prune_channel', self.key, prune_channels_multi_dims[1])
                # Use the mask to index the tensor
                weight = torch.index_select(self.weight, dim=extract_weight_dim, index=mask.nonzero().squeeze().to(self.weight.device))
                return weight
        
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
        nn.Linear.__init__(self, in_features, out_features)
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
        
        self.layer_type = 'linear'
        self.prune_name = kwargs['prune_name']
        self.prune_tgt = kwargs['prune_tgt']
        self.pruning_module = kwargs['pruning_module']
        self.key = kwargs['key']
        # if 'local' in self.prune_name and self.prune_tgt == 'weight':
        #     self.prune_weight(self.weight, self.layer_type)
    

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.prune_tgt == 'hidden_repr':
            # print('-----\n')
            # print('input_shape: ', x.shape)
            # print("prev weight.shape", self.weight.shape)
            input_dim = x.dim()
            input_shape = x.shape
            linear_layer_info = {
                'weight': self.weight.data,
            }
            pruned_h, pruned_dims, prune_channels_multi_dims = self.pruning_module.batch_pruning(x, self.layer_type, linear_layer_info, self.key)
            weight = self.extract_weight(input_dim, pruned_dims, prune_channels_multi_dims, self.layer_type)
            
            result = F.linear(pruned_h, transpose(weight, self.fan_in_fan_out), bias=self.bias)
        elif self.prune_tgt == 'weight':
            pruned_h = self.extract_input(x, self.layer_type)
            result = F.linear(pruned_h, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        self.pruning_module.cal_repr_distribution(pruned_h, 'pruned_hist')
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

        self.layer_type = 'conv2d'
        self.prune_name = kwargs['prune_name']
        self.prune_tgt = kwargs['prune_tgt']
        self.pruning_module = kwargs['pruning_module']
        self.key = kwargs['key']
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
            pruned_h, pruned_dims, prune_channels_multi_dims = self.pruning_module.batch_pruning(x, self.layer_type, conv2d_layer_info, self.key)
            weight = self.extract_weight(input_dim, pruned_dims, prune_channels_multi_dims)

            result = F.conv2d(
                pruned_h,
                weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.prune_tgt == 'weight':
            pruned_h = self.extract_input(x, self.layer_type)
            b = self.weight.shape
            result = F.conv2d(
                pruned_h,
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


class BasePruning:
    def __init__(self, cfg):
        self.prune_name = cfg['prune_name']
        self.prune_tgt = cfg['prune_tgt']
        self.prune_norm = cfg['prune_norm']
        self.prune_hyper = cfg['prune_hyper'] 
        self.prune_dim = cfg['prune_dim'] 
        self.prune_dim_select_mode = cfg['prune_dim_select_mode'] 
        self.batch_integ = cfg['batch_integ']   

        self.pq_p = cfg['pq_p']
        self.pq_q = cfg['pq_q']
        self.gamma = cfg['gamma']
        self.beta = cfg['beta']
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

    def reset_pruning_info(self):
        self.pruning_info = {}
        pass
    
    def update_pruning_info(self, info):
        self.pruning_info.update(info)
        pass
    
    def cal_repr_distribution(self, h, info_key):
        temp_h = h.detach().to(torch.float32).cpu().numpy()
        hist, _ = np.histogram(temp_h, bins=self.bin_edges)
        self.update_pruning_info({info_key: (hist/temp_h.shape[0]).tolist()})

class HiddenRepresentationPruning(BasePruning):

    def __init__(self, cfg, key):
        BasePruning.__init__(self, cfg)
        self.key = key
        

    def batch_pruning(self, h, layer_type, layer_info, key):
        # Exclude the first dimension (batch size) and the prune_dim
        # calculate the pq-index per sample
        exclude_dim = 0 if self.batch_integ in ['inter', 'union'] and h.dim() >= 3 else None
        prune_channels_multi_dims = [None] * h.dim()
        saving_flops_multi_dims = [0] * h.dim()
        h_shape = h.shape
        h_type = h.dtype
        
        self.cal_repr_distribution(h, f'{self.key}_vanilla_hist')
        if 'unstruct' in self.prune_name:
            if self.batch_integ in ['inter', 'union']:
                raise ValueError('Not valid batch integration method')
            if 'magunstruct' in self.prune_name:
                flattened_h = h.view(h.size(0), -1)
                norm_along_dim_1 = torch.linalg.vector_norm(flattened_h, ord=self.prune_norm, dim=1)
                _, sorted_indices = torch.sort(norm_along_dim_1, dim=1)
                num_indices_to_prune = int(self.prune_hyper * sorted_indices.size(1))
                # Select the indices to prune (lowest norms)
                prune_indices = sorted_indices[:, :num_indices_to_prune]
                mask = torch.ones_like(h, dtype=torch.bool)
                if prune_indices is not None:
                    # Mark the indices to be pruned as False
                    mask[:, prune_indices] = False
                pruned_h = h * mask.to(h.device)
                return pruned_h, None, None
            elif 'pqunstruct' in self.prune_name:
                pass
            
            elif 'wandaunstruct' in self.prune_name:
                if layer_type == 'linear':
                    dim = (0, 1)
                    h_norm = torch.linalg.vector_norm(h, ord=2, dim=dim)
                    h_norm = h_norm.view(1, -1)
                elif layer_type == 'conv2d':
                    raise ValueError('Not valid layer type conv2D')
                    # dim = (0, 2, 3)
                    # h_norm = torch.linalg.vector_norm(h, ord=2, dim=dim)
                    # h_norm = h_norm.view(1, -1, 1, 1)
                metric = layer_info['weight'].abs() * h_norm
                _, sorted_idx = torch.sort(metric, dim=1) 
                pruned_idx = sorted_idx[:,:int(layer_info['weight'].size(1) * self.prune_hyper)] 
                # layer_info['weight'].scatter_(dim=1, index=pruned_idx, src=0)
                # pruned_dims
                # prune_channels_multi_dims
                return h, 'wandaunstrcut', pruned_idx.to(h.device)
        elif 'struct' in self.prune_name:
            if self.prune_dim_select_mode == 'max':
                for prune_dim in self.prune_dim:
                    prune_dim = (h.dim() + prune_dim) % h.dim()
                    prune_channels = self.apply_pruning(h, key, prune_dim, exclude_dim)
                    prune_channels = self.apply_batch_integ(h_shape[prune_dim], prune_channels)
                    prune_channels_multi_dims[prune_dim] = prune_channels

                    saving_flops = self.cal_saving_flops(h, prune_dim, prune_channels, layer_type, layer_info)
                    saving_flops_multi_dims[prune_dim] = saving_flops

                if len(self.prune_dim) == 1:
                    final_prune_dim = (h.dim() + self.prune_dim[0]) % h.dim()
                else:
                    raise ValueError('Not valid prune dim')
                # final_prune_dim = np.argmax(saving_flops_multi_dims)
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
                self.update_pruning_info(cur_batch_info)
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
        # print('apply_pruning', prune_dim, h.dim(), h.shape)
        if prune_dim >= h.dim():
            raise ValueError('Not valid pruning dimension')
            # prune_dim = h.dim() - 1
        if exclude_dim is not None and exclude_dim != 0:
            raise ValueError('Not valid exclude dimension')
        # No pruning
        if self.prune_hyper == 9999:
            return [torch.empty(0)]

        if 'pqstruct' in self.prune_name:
            prune_channels = self.pq_struct(h, key, prune_dim, exclude_dim)
            return prune_channels
        elif 'magstruct' in self.prune_name:
            prune_channels = self.mag_struct(h, key, prune_dim, exclude_dim)
            return prune_channels
        # elif 'magunstruct' in self.prune_name:
        #     pruned_indices = self.mag_unstruct(h, key)
        #     return pruned_indices
        else:
            raise ValueError('Not valid pruning method')
        

    def pq_struct(self, h, key, prune_dim, exclude_dim):
        calc_dim = 0
        if exclude_dim != None and exclude_dim == 0:
            calc_dim = 1
 
        dims_to_aggregate = tuple(i for i in range(h.dim()) if i != prune_dim and i != exclude_dim)
        norm_across_other_dims = torch.linalg.vector_norm(h, ord=self.prune_norm, dim=dims_to_aggregate)     

        # print('norm_across_other_dims', norm_across_other_dims.shape, norm_across_other_dims.dim())
        norm_p = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_p, dim=calc_dim)
        norm_q = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_p, dim=calc_dim) + 1e-10
        
        dimension = h.shape[prune_dim]
        pq_indices = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * norm_p / norm_q)

        # add additional dimension if dimension is 0
        if pq_indices.dim() == 0:
            pq_indices = pq_indices.unsqueeze(0)

        if torch.isnan(pq_indices).any():
            raise ValueError('pq_indices contains nan values')

        lower_bound = dimension * (1 + self.eta) ** (-self.pq_q / (self.pq_q - self.pq_p)) * (1 - pq_indices) ** (self.pq_q * self.pq_p / (self.pq_q - self.pq_p))
        beta_tensor = torch.full_like(lower_bound, self.beta)
        prune_channels_count = torch.floor(dimension * torch.min(self.gamma * (1 - lower_bound / dimension), beta_tensor))

        _, sorted_channels = torch.sort(norm_across_other_dims, dim=calc_dim)
        # print('sorted_channels', sorted_channels.shape, sorted_channels, prune_channels_count)
        if sorted_channels.dim() > 1:
            prune_channels = [sorted_channels[i, :int(count.item())] for i, count in enumerate(prune_channels_count)]
        else:
            prune_channels = [sorted_channels[:int(prune_channels_count[0].item())]]

        info = {
            f"{key}_norm_across_other_dims": norm_across_other_dims.mean(dim=0).squeeze(0).tolist(),
            f"{key}_pq_indices": pq_indices.mean(dim=0).squeeze(0).tolist(),
        }
        self.update_pruning_info(info)
        return prune_channels

    # def mag_unstruct(self, h):
    #     flattened_h = h.view(h.size(0), -1)
    #     norm_along_dim_1 = torch.linalg.vector_norm(flattened_h, ord=self.prune_norm, dim=1)
    #     _, sorted_indices = torch.sort(norm_along_dim_1, dim=1)
    #     num_indices_to_prune = int(self.prune_hyper * sorted_indices.size(1))
    #     # Select the indices to prune (lowest norms)
    #     pruned_indices = sorted_indices[:, :num_indices_to_prune]
    #     return pruned_indices
    
    def mag_struct(self, h, key, prune_dim, exclude_dim):
        calc_dim = 0
        if exclude_dim != None and exclude_dim == 0:
            calc_dim = 1
        dims_to_aggregate = tuple(i for i in range(h.dim()) if i != prune_dim and i != exclude_dim)
        norm_across_other_dims = torch.linalg.vector_norm(h, ord=self.prune_norm, dim=dims_to_aggregate)        
        prune_channels_count = int(self.prune_hyper * h.shape[prune_dim])
        _, sorted_channels = torch.sort(norm_across_other_dims, dim=calc_dim)
        if sorted_channels.dim() > 1:
            prune_channels = [sorted_channels[i, :prune_channels_count] for i in range(sorted_channels.size(0))]
        else:
            prune_channels = [sorted_channels[:prune_channels_count]]
        return prune_channels

        

class WeightPruning(BasePruning):

    def __init__(self, cfg, key):
        BasePruning.__init__(self, cfg)
        self.key = key
        self.target_modules = _get_target_modules(cfg)
        pass
    
    def global_pruning(self, model):
        if self.prune_hyper == 9999:
            return
        
        if 'struct' in self.prune_name:
            all_weights = []
            # TODO, prune_dim can only has 1 dim now
            channel_norms = []
            for name, module in model.named_modules():
                if _check_target_module_exists(self.target_modules, name):
                    if hasattr(module, 'weight') and module.weight is not None:
                        dims_to_aggregate = tuple(i for i in range(module.weight.data.dim()) if i != self.prune_dim[0])
                        norm_across_other_dims = torch.linalg.vector_norm(module.weight.data, ord=self.prune_norm, dim=dims_to_aggregate) 
                        for i, norm in enumerate(norm_across_other_dims):
                            channel_norms.append((norm.item(), name, i, module.weight.shape))  # Store norm, layer name, channel index, and shape

            if 'magstruct' in self.prune_name:
                channel_norms.sort(key=lambda x: x[0])
                num_channels_to_prune = int(len(channel_norms) * self.prune_hyper) 
                pruning_threshold = channel_norms[num_channels_to_prune - 1][0] if num_channels_to_prune > 0 else float('inf')
            elif 'pqstruct' in self.prune_name:
                norm_p = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_p, dim=0)
                norm_q = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_p, dim=0) + 1e-10
                
                dimension = len(channel_norms)
                pq_indices = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * norm_p / norm_q)

                # add additional dimension if dimension is 0
                if pq_indices.dim() == 0:
                    pq_indices = pq_indices.unsqueeze(0)

                if torch.isnan(pq_indices).any():
                    raise ValueError('pq_indices contains nan values')

                lower_bound = dimension * (1 + self.eta) ** (-self.pq_q / (self.pq_q - self.pq_p)) * (1 - pq_indices) ** (self.pq_q * self.pq_p / (self.pq_q - self.pq_p))
                beta_tensor = torch.full_like(lower_bound, self.beta)
                prune_channels_count = torch.floor(dimension * torch.min(self.gamma * (1 - lower_bound / dimension), beta_tensor))
                pruning_threshold = channel_norms[prune_channels_count - 1][0] if num_channels_to_prune > 0 else float('inf')

            pruning_info = collections.defaultdict(list)
            for norm_value, layer_name, channel_index, shape in channel_norms:
                if norm_value > pruning_threshold:
                    break  # Only consider channels below the threshold
                pruning_info[layer_name].append(channel_index)

            # Step 2: Apply Pruning
            for layer_name, channel_index_list in pruning_info.items():
                layer = dict(model.named_modules())[layer_name]
                with torch.no_grad():
                    # Prune the specific channel
                    layer.weight.data = self.prune_w(layer.weight.data, self.prune_dim[0], channel_index_list)
                    # Store the pruned channel index if needed
                    layer.input_prune_channels = channel_index_list

        elif 'unstruct' in self.prune_name:
            all_weights = []
            for name, module in model.named_modules():
                if _check_target_module_exists(self.target_modules, name):
                    if hasattr(module, 'weight') and module.weight is not None:
                        all_weights.append(module.weight.data.view(-1))

            if 'magunstruct' in self.prune_name:
                # Concatenate all weights and convert to a single vector
                all_weights_vector = torch.cat(all_weights)
                # Rank weights by absolute value and find the threshold for pruning
                num_weights_to_prune = int(all_weights_vector.numel() * self.prune_hyper) 
                threshold = torch.kthvalue(torch.abs(all_weights_vector), num_weights_to_prune).values
            elif 'pqunstruct' in self.prune_name:
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
                beta_tensor = torch.full_like(lower_bound, self.beta)
                prune_channels_count = torch.floor(dimension * torch.min(self.gamma * (1 - lower_bound / dimension), beta_tensor))
                threshold = torch.kthvalue(torch.abs(all_weights_vector), prune_channels_count).values


            # Apply pruning directly without a separate mask
            index = 0
            for name, module in model.named_modules():
                if _check_target_module_exists(self.target_modules, name):
                    if hasattr(module, 'weight') and module.weight is not None:
                        numel = module.weight.data.numel()
                        module.weight.data.view(-1).abs_().clamp_(min=threshold)
                        index += numel

    def local_pruning(self, w, layer_type, layer_info, key, h = None):
        if self.prune_hyper == 9999:
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

                return pruned_w, pruned_dims, prune_channels_multi_dims
        else:
            raise ValueError('Not valid pruning method')
    
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
        norm_across_other_dims = torch.linalg.vector_norm(w, ord=self.prune_norm, dim=dims_to_aggregate)     

        norm_p = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_p, dim=calc_dim)
        norm_q = torch.linalg.vector_norm(norm_across_other_dims, ord=self.pq_p, dim=calc_dim) + 1e-10
        
        dimension = w.shape[prune_dim]
        pq_indices = (1 - dimension ** (1/self.pq_q - 1/self.pq_p) * norm_p / norm_q)

        # add additional dimension if dimension is 0
        if pq_indices.dim() == 0:
            pq_indices = pq_indices.unsqueeze(0)

        if torch.isnan(pq_indices).any():
            raise ValueError('pq_indices contains nan values')

        lower_bound = dimension * (1 + self.eta) ** (-self.pq_q / (self.pq_q - self.pq_p)) * (1 - pq_indices) ** (self.pq_q * self.pq_p / (self.pq_q - self.pq_p))
        beta_tensor = torch.full_like(lower_bound, self.beta)
        prune_channels_count = torch.floor(dimension * torch.min(self.gamma * (1 - lower_bound / dimension), beta_tensor))

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
        norm_across_other_dims = torch.linalg.vector_norm(w, ord=self.prune_norm, dim=dims_to_aggregate)        
        _, sorted_channels = torch.sort(norm_across_other_dims, dim=0)
        prune_channels_count = int(self.prune_hyper * w.shape[prune_dim])
        prune_channels = sorted_channels[:int(prune_channels_count)]
        return prune_channels




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