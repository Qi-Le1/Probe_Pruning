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
from module import to_device, TRANSFORMERS_MODELS_TO_EWI_TARGET_MODULES_MAPPING
from functools import reduce

class EwiModel(torch.nn.Module):
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
        
        kwargs = {
            "prune_tgt": cfg['prune_tgt'],
            "prune_metric": cfg['prune_metric'],
            "key": key,
            "fan_in_fan_out": False,
            "device": target.weight.device,
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
            
            new_module = self._create_new_module(prune_name, target, key)
            self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        fan_in_fan_out = getattr(new_module, "fan_in_fan_out", False)
        print('fan_in_fan_out', child_name, fan_in_fan_out)
        # if fan_in_fan_out is True, the layer is conv1d layer in GPT2
        # which is a self-defined layer, not the traditional Conv1D
        # print('old_module.weight', old_module.weight)
        new_module.weight = transpose(old_module.weight, fan_in_fan_out)
        new_module.weight.requires_grad = False
        new_module.device = old_module.weight.device
        if hasattr(old_module, "bias"):
            # if old_module.bias is not None:
            # print('old_module.bias', old_module.bias)
            # print('old_module.bias.shape', old_module.bias.shape)
            # print('new_module.bias', new_module.bias)
            # print('new_module.bias.shape', new_module.bias.shape)
            
            new_module.bias = old_module.bias

        # if getattr(old_module, "state", None) is not None:
        #     new_module.state = old_module.state
        #     new_module.to(old_module.weight.device)

        # # dispatch to correct device
        # for name, module in new_module.named_modules():
        #     if "pruner_" in name:
        #         module.to(old_module.weight.device)


    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

def transpose(weight, fan_in_fan_out):
    transposed_weight = weight.T if fan_in_fan_out else weight
    return nn.Parameter(transposed_weight)
    # return transposed_weight

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
    target_modules = TRANSFORMERS_MODELS_TO_EWI_TARGET_MODULES_MAPPING[cfg['model_type']]
    if 'cust_tgt_modules' in cfg and 'default' not in cfg['cust_tgt_modules']:
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

class EwiLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.prune_tgt = kwargs['prune_tgt']
        self.prune_metric = kwargs['prune_metric']
        self.key = kwargs['key']

        self.pruning_channel_ratio = []
        self.input_prune_channels = None
        self.weight_norm_across_channel_dims = None
        return
    

class Linear(nn.Linear, EwiLayer):
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
        EwiLayer.__init__(self, in_features=in_features, out_features=out_features, **kwargs)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        
        self.layer_type = 'linear'

        self.out_dim = out_features
        self.in_dim = in_features
        self.prune_metric = cfg['prune_metric']
        self.nsamples = 0
        self.device = kwargs['device']

        self.baseline_inp = torch.zeros((self.in_dim), device=self.device)
        if self.prune_metric == "WIFN":
            self.scaler_inp = torch.zeros((self.in_dim), device=self.device)
        elif self.prune_metric == "IFV" or self.prune_metric == "WIFV":
            self.fluc_inp = torch.zeros((self.in_dim), device=self.device)
        else:
            raise ValueError(f"Unknown pruning method {self.prune_name}")
        # if self.prune_name == "wanda-sp":
        #     self.scaler_row = torch.zeros((self.columns), device=self.device)
        # elif self.prune_name == "flap":
            
        # elif self.prune_name == "pq-nobias" or self.prune_name == "pq-bias":
        #     self.baseline_inp = torch.zeros((self.in_dim), device=self.device)
        #     if self.prune_metric == "WIFN":
        #         self.scaler_inp = torch.zeros((self.in_dim), device=self.device)
        #     elif self.prune_metric == "IFV" or self.prune_metric == "WIFV":
        #         self.fluc_inp = torch.zeros((self.in_dim), device=self.device)
        
        
    def get_pre_hook(self):
        # if self.prune_name == "wanda-sp":
        #     def add_batch(inp, out):
        #         if len(inp.shape) == 2:
        #             inp = inp.unsqueeze(0)
        #         tmp = inp.shape[0]
        #         # if isinstance(self.layer, nn.Linear):
        #         if len(inp.shape) == 3:
        #             inp = inp.reshape((-1, inp.shape[-1]))
        #         inp = inp.t()
                
        #         self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        #         self.nsamples += tmp

        #         inp = inp.type(torch.float32)

        #         self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        # elif self.prune_name == "flap":
        #     def add_batch(inp, out):
        #         if len(inp.shape) == 2:
        #             inp = inp.unsqueeze(0)
        #         batch_size = inp.shape[0]
        #         # if isinstance(self.layer, nn.Linear):
        #         if len(inp.shape) == 3:
        #             inp = inp.reshape((-1, inp.shape[-1]))
        #         inp = inp.t()   # (dim, seqlen * batch_size)

        #         old_baseline_inp = self.baseline_inp
        #         self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        #         self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        #         if self.prune_metric == "WIFN":
        #             inp = inp.type(torch.float32)
        #             self.scaler_inp *= self.nsamples / (self.nsamples + batch_size)
        #             self.scaler_inp += torch.norm(inp, p=2, dim=1) ** 2  / (self.nsamples + batch_size)
        #         else:
        #             if self.nsamples == 0:
        #                 self.fluc_inp = 0
        #             else:
        #                 self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
        #                 self.fluc_inp += torch.sum((inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1) / (self.nsamples + batch_size)   # a²+b²+c²...没开根号

        #         self.nsamples += batch_size
        # elif self.prune_name == "pq-nobias" or self.prune_name == "pq-bias":
        def add_batch(inp, out):
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            batch_size = inp.shape[0]
            # if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   # (dim, seqlen * batch_size)

            old_baseline_inp = self.baseline_inp
            self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
            self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
            if self.prune_metric == "WIFN":
                inp = inp.type(torch.float32)
                self.scaler_inp *= self.nsamples / (self.nsamples + batch_size)
                self.scaler_inp += torch.norm(inp, p=2, dim=1) ** 2  / (self.nsamples + batch_size)
                # print('self.scaler_inp', self.scaler_inp)
            elif self.prune_metric == "IFV" or self.prune_metric == "WIFV":
                if self.nsamples == 0:
                    self.fluc_inp = 0
                else:
                    self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
                    self.fluc_inp += torch.sum((inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1) / (self.nsamples + batch_size)   # a²+b²+c²...没开根号
                    # print('inp', inp)
                    # print('old_baseline_inp', old_baseline_inp)
                    # print('self.baseline_inp: ', self.baseline_inp)
                    # print('self.fluc_inp: ', self.fluc_inp)
                    # print('self.baseline_inp: ', self.baseline_inp)
                    # print('self.nsamples', self.nsamples)
                    # print('coeff', (self.nsamples - 1) / (self.nsamples + batch_size - 1))

            self.nsamples += batch_size

        return add_batch

    def free(self):
        if hasattr(self, 'baseline_inp'):
            self.baseline_inp = None
        if hasattr(self, 'fluc_inp'):
            self.fluc_inp = None
        if hasattr(self, 'scaler_inp'):
            self.scaler_inp = None
        if hasattr(self, 'scaler_row'):
            self.scaler_row = None
        torch.cuda.empty_cache()  

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        result = F.linear(x, self.weight, bias=self.bias)
        
        # self.pruning_module.cal_repr_distribution(pruned_h, f'{self.key}_pruned_hist')
        result = result.to(previous_dtype)
        return result
    

# class Conv2d(nn.Conv2d, EwiLayer):
#     def __init__(
#         self,
#         prune_name,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: Union[int, Tuple[int]],
#         stride: Union[int, Tuple[int]] = 1,
#         padding: Union[int, Tuple[int]] = 0,
#         dilation: Union[int, Tuple[int]] = 1,
#         groups: int = 1,
#         **kwargs,
#     ):
#         nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
#         EwiLayer.__init__(
#             self,
#             in_features=in_channels,
#             out_features=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             **kwargs
#         )
#         # Freezing the pre-trained weight matrix
#         self.weight.requires_grad = False   

#         self.layer_type = 'conv2d'
#         # if 'local' in self.prune_name and self.prune_tgt == 'weight':
#         #     self.prune_weight(self.weight, self.layer_type)
            
#     def forward(self, x: torch.Tensor):
#         # print('input_shape: ', x.shape)
#         # print('pruned_conv2d')
#         previous_dtype = x.dtype
#         if self.prune_tgt == 'hidden_repr':
#             # print('-----\n')
#             # print('input_shape: ', x.shape)
#             # print("prev weight.shape", self.weight.shape)
#             input_dim = x.dim()
#             conv2d_layer_info = {
#                 'weight': self.weight.data,
#             }
#             pruned_h, pruned_dims, prune_channels_multi_dims = self.pruning_module.batch_pruning(x, self.layer_type, conv2d_layer_info, self.key)
#             weight = self.extract_weight(input_dim, pruned_dims, prune_channels_multi_dims, self.layer_type)

#             result = F.conv2d(
#                 pruned_h,
#                 weight,
#                 bias=self.bias,
#                 stride=self.stride,
#                 padding=self.padding,
#                 dilation=self.dilation,
#                 groups=self.groups,
#             )
#         elif self.prune_tgt == 'weight':
#             pruned_h = self.extract_input(x, self.layer_type)
#             # b = self.weight.shape
#             result = F.conv2d(
#                 pruned_h,
#                 self.weight,
#                 bias=self.bias,
#                 stride=self.stride,
#                 padding=self.padding,
#                 dilation=self.dilation,
#                 groups=self.groups,
#             )
#         # self.pruning_module.cal_repr_distribution(pruned_h, f'{self.key}_pruned_hist')
#         result = result.to(previous_dtype)
#         return result

