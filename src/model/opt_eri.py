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

class OPTEriModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.add_pruner(cfg['prune_name'])

    def add_pruner(self, prune_name):
        self._find_and_replace(prune_name)
        mark_no_trainable(self.model)        
        return
    
    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

    def _create_new_module(self, prune_name, target, key):
        bias = hasattr(target, "bias") and target.bias is not None
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        in_features = getattr(target, 'in_features', None)
        out_features = getattr(target, 'out_features', None)
        pruning_module = HiddenRepresentationPruning(cfg, key,target.weight.device, in_features, out_features)
        
        kwargs = {
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
            new_module.is_prune_out_dim = False
            if 'probe' in cfg['prune_metric']:
                # new_module.is_prune_out_dim = False
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
        # if hasattr(old_module, "bias"):
        #     # old_module might not have bias, bias=None
        #     # need to write into new_module, otherwise
        #     # the parent class will assign bias
        #     print('old_module.bias', old_module.bias)
        #     new_module.bias = old_module.bias

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
        self.cal_total_flops = True

        self.prune_metric = kwargs['prune_metric']
        self.pruning_module = kwargs['pruning_module']
        self.key = kwargs['key']

        self.pruning_channel_ratio = []
        self.input_prune_channels = None
        self.weight_norm_across_channel_dims = None
        pass
        
    def extract_in_weight(self, input_dim, pruned_dim, preserve_channels, layer_type):
        # unstruct pruning
        if pruned_dim is None:
            return self.weight
        
        if layer_type == 'linear':
            #  [batch_size, seq_lens, token_lens]
            if input_dim != 3:
                raise ValueError('Not valid input dimension')
            
            if pruned_dim != input_dim - 1:
                raise ValueError('Not valid input dimension')

            weight = torch.index_select(self.weight, dim=1, index=preserve_channels.to(self.weight.device))
            return weight
           
        
    def extract_out_weight(self, input_dim, pruned_dim, preserve_channels, layer_type):
        
        if layer_type == 'linear':
            #  [batch_size, seq_lens, token_lens]
            if input_dim != 3:
                raise ValueError('Not valid input dimension')
                
            weight = torch.index_select(self.weight, dim=0, index=preserve_channels.to(self.weight.device))
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
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.layer_type = 'linear'
        self.prune_metric = cfg['prune_metric']
    
    def check_fill_case(self):
        if cfg['qk_proj_prune'] == 'fill' and ('q_proj' in self.key or 'k_proj' in self.key):
            return True
        elif cfg['vo_proj_prune'] == 'fill' and ('v_proj' in self.key or 'o_proj' in self.key):
            return True
        return False

    # no bias in llama-2
    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            previous_dtype = x.dtype
            if 'probe' in cfg['prune_name'] and 'cal_mlp_probe_out_dim_metric' in kwargs and kwargs['cal_mlp_probe_out_dim_metric'] == True:
                # broadcast and return out_dim * in_dim matrix
                if 'keepseq' in cfg['prune_metric']:
                    return F.linear(x, self.weight)
                return x * self.weight
            elif 'probe' in cfg['prune_name'] and 'cal_attn_probe_out_dim_metric' in kwargs and kwargs['cal_attn_probe_out_dim_metric'] == True:
                # need to save s dimension for the following probe                    
                result = F.linear(x, self.weight)
                result = result.to(previous_dtype)
                return result
            elif 'probe' in cfg['prune_name'] and 'cal_attn_probe_out_dim_metric_compressseq' in kwargs and kwargs['cal_probe_out_dim_metric_compressseq'] == True:
                return x * self.weight
            # print('-----\n')
            # print('input_shape: ', x.shape)
            # print("prev weight.shape", self.weight.shape)
            batch_size = x.size(0)
            input_dim = x.dim()
            seq_len = x.size(1)
            input_shape = x.shape

            linear_layer_info = {
                'weight': self.weight.data,
            }

            # enter down-proj after delete outputdim of gate-proj and up-proj
            # enter o-proj after attention
            if 'probe' in cfg['prune_name'] and self.is_prune_out_dim == False:
                # print('here1')
                if 'attn' in self.key:
                    # if 'probe' in cfg['prune_name']:
                        # res_original = F.linear(x, self.weight)

                    if self.check_fill_case():
                        x = x[..., kwargs['probe_out_dim_indices']]
                    # if it is probe and parallel, we consider all the structure and operations
                    # we already know the weight index to extract
                    weight = torch.index_select(self.weight, dim=1, index=kwargs['probe_out_dim_indices'].to(self.weight.device))
                        # print('attnprobemix, probe_out_dim_indices', kwargs['probe_out_dim_indices'])

                        # res_after_extract = F.linear(x, weight)
                        # print('res_original', res_original)
                        # print('res_after_extract', res_after_extract)
                        
                    # else:
                        # prune out dim situation
                        # but not probe (only consider current layer's input and weight)
                        # or probe, but do not consider parallel strucutre's effect, only considering gate or up individually.
                        # non_zero_indices = torch.nonzero(torch.sum(x, dim=(0,1)), as_tuple=True)[-1]
                        # weight = torch.index_select(self.weight, dim=1, index=non_zero_indices.to(self.weight.device))
                        # x = x[..., non_zero_indices]

                else:
                    # if 'probe' in cfg['prune_name']:
                        # if it is probe and parallel, we consider all the structure and operations
                        # we already know the weight index to extract
                    weight = torch.index_select(self.weight, dim=1, index=kwargs['probe_out_dim_indices'].to(self.weight.device))
                    # else:
                    #     # prune out dim situation
                    #     # but not probe (only consider current layer's input and weight)
                    #     # or probe, but do not consider parallel strucutre's effect, only considering gate or up individually.
                    #     non_zero_indices = torch.nonzero(torch.sum(x, dim=(0,1)), as_tuple=True)[-1]
                    #     weight = torch.index_select(self.weight, dim=1, index=non_zero_indices.to(self.weight.device))
                    #     x = x[..., non_zero_indices]
            # for probe situation, entering up-proj or gate-proj
            # or entering q/k/v-proj
            elif 'probe' in cfg['prune_name'] and 'probe_out_dim_indices' in kwargs:
                # # weight dim 1 should have same size as original h dim 2
                # mask = torch.ones(self.weight.size(0), dtype=torch.bool)
                # # Mark the indices to be pruned as False
                # mask[kwargs['probe_out_dim_indices']] = False
                
                # Use the mask to index the tensor
                # print('here2')
                if 'attn' in self.key:
                    weight = torch.index_select(self.weight, dim=0, index=kwargs['probe_out_dim_indices'].to(self.weight.device))
                    # print('weight.shape', weight.shape)
                    # record to fill zero
                    # if 'para' not in cfg['prune_name']:
                    #     self.out_selected_dim = kwargs['probe_out_dim_indices']
                    if self.check_fill_case():
                        self.out_selected_dim = kwargs['probe_out_dim_indices']

                    # print('attn probe_out_dim_indices')
                else:
                    weight = torch.index_select(self.weight, dim=0, index=kwargs['probe_out_dim_indices'].to(self.weight.device))
                    # print('weight.shape', weight.shape)
                    # record to fill zero
                    # if 'para' not in cfg['prune_name']:
                    #     self.out_selected_dim = kwargs['probe_out_dim_indices']
            else:
                # print('here3')
                x, pruned_dim, preserve_channels = self.pruning_module.batch_pruning(x, self.layer_type, linear_layer_info, self.key, self.is_prune_out_dim)
                # print('xafter prune', x)
                # if 'WO' in cfg['prune_metric']:
                #     weight = self.extract_out_weight(input_dim, pruned_dim, preserve_channels, self.layer_type)
                # else:
                weight = self.extract_in_weight(input_dim, pruned_dim, preserve_channels, self.layer_type)
            
            if 'flap' in cfg['prune_metric']:
                # all_channels = torch.arange(self.in_features, dtype=preserve_channels.dtype, device=preserve_channels.device)
                # mask = all_channels[preserve_channels]
                # mean_x = torch.mean(x, dim=1)
                # flap = (x - mean_x) * 
                # output_bias = ((mean_inp * ~mask.to(self.weight.data.device)) @ self.weight.data.T)
                # result = F.linear(x, weight, bias=output_bias)
                pass
            else:
                result = F.linear(x, weight)
            # if 'o_proj' in self.key:
            #     print('result', result, flush=True)
            # refill the pruned output dim with 0
            # gate-proj & up-proj
            # q/k/v-proj
            if 'probe' in cfg['prune_name'] and self.is_prune_out_dim == True:
                # dont need to refill 0 because all indices are determined
                if 'attn' in self.key:
                    
                    if self.check_fill_case():
                        refilling_output = torch.zeros(batch_size, seq_len, self.out_features, device=result.device, dtype=result.dtype)
                        # Insert the output from the pruned layer into the correct positions in the extended tensor
                        # print('extended_output.shape', extended_output.shape)
                        # print('key', self.key)
                        # print('self.out_selected_dim', torch.nonzero(self.out_selected_dim).squeeze())
                        # print('gate&up', (self.out_selected_dim.shape[0] - torch.sum(self.out_selected_dim))/self.weight.shape[0], self.key)
                        refilling_output[..., self.out_selected_dim] = result

                        # 'extended_output' now has the pruned outputs filled in and zeros elsewhere
                        result = refilling_output
                            # print('result.shape', result.shape)
                        # else:
                        #     pass
                    # else:
                    #     refilling_output = torch.zeros(batch_size, seq_len, self.out_features, device=result.device, dtype=result.dtype)
                    #     # Insert the output from the pruned layer into the correct positions in the extended tensor
                    #     # print('extended_output.shape', extended_output.shape)
                    #     # print('key', self.key)
                    #     # print('self.out_selected_dim', torch.nonzero(self.out_selected_dim).squeeze())
                    #     # print('gate&up', (self.out_selected_dim.shape[0] - torch.sum(self.out_selected_dim))/self.weight.shape[0], self.key)
                    #     refilling_output[..., self.out_selected_dim] = result

                    #     # 'extended_output' now has the pruned outputs filled in and zeros elsewhere
                    #     result = refilling_output
                # else:
                #     if 'probe' in cfg['prune_name']:
                #         pass
                #     else:
                #         refilling_output = torch.zeros(batch_size, seq_len, self.out_features, device=result.device, dtype=result.dtype)
                #         # Insert the output from the pruned layer into the correct positions in the extended tensor
                #         # print('extended_output.shape', extended_output.shape)
                #         # print('key', self.key)
                #         # print('self.out_selected_dim', torch.nonzero(self.out_selected_dim).squeeze())
                #         # print('gate&up', (self.out_selected_dim.shape[0] - torch.sum(self.out_selected_dim))/self.weight.shape[0], self.key)
                #         refilling_output[..., self.out_selected_dim] = result

                #         # 'extended_output' now has the pruned outputs filled in and zeros elsewhere
                #         result = refilling_output
            # elif self.prune_tgt == 'weight':
            #     pruned_h = self.extract_input(x, self.layer_type)
            #     result = F.linear(pruned_h, self.weight, bias=self.bias)
            
            # self.pruning_module.cal_repr_distribution(pruned_h, f'{self.key}_pruned_hist')
            result = result.to(previous_dtype)
            # print('result.shape: ', result.shape, self.key)
        return result