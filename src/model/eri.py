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
        if cfg['prune_tgt'] == 'hidden_repr':
            self.pruning_module = HiddenRepresentationPruning(cfg)
        elif cfg['prune_tgt'] == 'weight':
            self.pruning_module = WeightPruning(cfg)
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
            target_module_found = any(key.endswith(target_key) for target_key in target_modules)

        return target_module_found

    def _create_new_module(self, pruner_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        kwargs = {
            "pruning_module": self.pruning_module,
            "fan_in_fan_out": False,
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
            new_module = Conv2d(pruner_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
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
        print('key_list: ', key_list)
        target_modules = TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING[cfg['model_name']]
        print('target_modules: ', target_modules)
        for key in key_list:
            if not self._check_target_module_exists(target_modules, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            
            new_module = self._create_new_module(pruner_name, target)
            self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING[cfg['model_name']]} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
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
        pass
    
    def extract_weight(self, input_dim, pruned_h, pruned_dims, prune_channels_multi_dims):
        # unstruct pruning
        if pruned_dims is None:
            return self.weight
        
        # [batch_size, in_features]
        if input_dim == 2:
            check_dim = 1
        # [batch_size, seq_lens, token_lens]
        elif input_dim == 3:
            check_dim = 2
        else:
            raise ValueError('Not valid input dimension')
        
        if pruned_dims[check_dim] is None or prune_channels_multi_dims[check_dim] is None:
            return self.weight
        else:
            # weight dim 1 should have same size as original h dim 2
            mask = torch.ones(self.weight.size(1), dtype=torch.bool)
            # Mark the indices to be pruned as False
            mask[prune_channels_multi_dims[check_dim]] = False
            # Use the mask to index the tensor
            weight = torch.index_select(self.weight, dim=1, index=mask.nonzero().squeeze())
            return weight

    
class Linear(nn.Linear, EriLayer):
    def __init__(
        self,
        pruner_name,
        in_features,
        out_features,
        pruning_module,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        EriLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.pruning_module = pruning_module
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        print('x.shape: ', x.shape)
        input_dim = x.dim()
        pruned_h, pruned_dims, prune_channels_multi_dims = self.pruning_module.batch_pruning(x, 'linear')
        weight = self.extract_weight(input_dim, pruned_h, pruned_dims, prune_channels_multi_dims)
        print(pruned_h.shape, weight.shape, self.weight.shape)
        result = F.linear(pruned_h, transpose(weight, self.fan_in_fan_out), bias=self.bias)
        result = result.to(previous_dtype)
        return result
    

class Conv2d(nn.Conv2d, EriLayer):
    def __init__(
        self,
        pruner_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding)
        EriLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.pruning_module = kwargs["pruning_module"]

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        result = F.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
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



class HiddenRepresentationPruning(BasePruning):

    def __init__(self, cfg):
        BasePruning.__init__(self, cfg)
        pass

    def batch_pruning(self, h, layer_type):
        # Exclude the first dimension (batch size) and the prune_dim
        # calculate the pq-index per sample
        if self.batch_integ in ['inter', 'union']:
            exclude_dim = 0

        prune_channels_multi_dims = [None for _ in range(h.dim())]
        prune_eles_multi_dims = [0 for _ in range(h.dim())]
        h_shape = h.shape
        # prune_channels_multi_dims = []
        # prune_eles_multi_dims = []
        if self.prune_name == 'base-unstruct':
            pruned_h = self.apply_pruning(h)
            return pruned_h, None, None
        
        
        elif self.prune_name in ['pq', 'base-struct']:
            if self.prune_dim_select_mode == 'max':
                for prune_dim in self.prune_dim:
                    prune_channels = self.apply_pruning(h, prune_dim, exclude_dim)
                    prune_channels = self.apply_batch_integ(h_shape[prune_dim], prune_channels)
                    prune_channels_multi_dims[prune_dim] = prune_channels

                    # prune_eles = self.cal_prune_eles(h, prune_dim, prune_channels)
                    prune_eles = len(prune_channels)
                    prune_eles_multi_dims[prune_dim] = prune_eles

                final_prune_dim = np.argmax(prune_eles_multi_dims)
                pruned_h = self.prune_h(h, final_prune_dim, prune_channels_multi_dims[final_prune_dim])

                pruned_dims = [final_prune_dim if dim == final_prune_dim else None for dim in range(h.dim())]
                for dim in range(len(prune_channels_multi_dims)):
                    if dim != final_prune_dim:
                        prune_channels_multi_dims[dim] = None
                        prune_eles_multi_dims[dim] = 0
                        
                return pruned_h, pruned_dims, prune_channels_multi_dims
            elif self.prune_dim_select_mode == 'casc':
                for prune_dim in self.prune_dim:
                    prune_channels = self.apply_pruning(h, prune_dim, exclude_dim)
                    prune_channels = self.apply_batch_integ(h_shape[prune_dim], prune_channels)
                    prune_channels_multi_dims[prune_dim] = prune_channels
                    
                    prune_eles = self.cal_prune_eles(h, prune_dim, prune_channels)
                    prune_eles_multi_dims[prune_dim] = prune_eles

                    pruned_h = self.prune_h(h, prune_dim, prune_channels_multi_dims[prune_dim])
                    h = pruned_h
                pruned_dims = [dim if dim in self.prune_dim else None for dim in range(h.dim())]
                return pruned_h, pruned_dims, prune_channels_multi_dims
    
    def prune_h(self, h, prune_dim, prune_channels):
        # Create a boolean mask for all indices
        mask = torch.ones(h.size(prune_dim), dtype=torch.bool)
        # Mark the indices to be pruned as False
        mask[prune_channels] = False
        # Use the mask to index the tensor
        pruned_h = h.index_select(dim=prune_dim, index=mask.nonzero().squeeze())
        return pruned_h

    def cal_prune_eles(self, h, prune_dim, prune_channels):
        if prune_channels is None:
            return 0
        rest_dims = tuple(i for i in range(h.dim()) if i != prune_dim)
        prune_eles = len(prune_channels) * reduce(lambda x, y: x * y, h.shape[rest_dims])
        return prune_eles
    

    def apply_batch_integ(self, cur_total_channels, prune_channels):
        if self.batch_integ == 'inter':
            sets = [set(tensor.tolist()) for tensor in prune_channels]
            if len(sets) == 0:
                sets = [set()]
            intersected_set = set.intersection(*sets)
            prune_channels = torch.tensor(list(intersected_set))
        elif self.batch_integ == 'union':
            sets = [set(tensor.tolist()) for tensor in prune_channels]
            if len(sets) == 0:
                sets = [set()]
            intersected_set = set.union(*sets)
            prune_channels = torch.tensor(list(intersected_set))
        elif self.batch_integ == 'full':
            prune_channels = torch.tensor(prune_channels)
        else:
            raise ValueError('Not valid batch integration method')
        if prune_channels.numel() == 0:
            return None
        if prune_channels.numel() >= cur_total_channels:
            prune_channels_list = prune_channels.tolist()
            prune_channels_list.remove(random.choice(prune_channels_list))
            # Convert back to tensor
            prune_channels = torch.tensor(prune_channels_list, dtype=prune_channels.dtype)
            warnings.warn("Attempting to prune all channels. Keeping one channel for calculation.")
        return prune_channels
    
    def apply_pruning(self, h, prune_dim=None, exclude_dim=None):
        if prune_dim >= h.dim():
            raise ValueError('Not valid pruning dimension')
        if exclude_dim is not None and exclude_dim != 0:
            raise ValueError('Not valid exclude dimension')
        
        if self.prune_name == 'pq':
            prune_channels = self.pq(h, prune_dim, exclude_dim)
            return prune_channels
        elif self.prune_name == 'base-struct':
            prune_channels = self.base_struct(h, prune_dim, exclude_dim)
            return prune_channels
        elif self.prune_name == 'base-unstruct':
            pruned_indices = self.base_unstruct(h)
            return pruned_indices
        else:
            raise ValueError('Not valid pruning method')
        

    def pq(self, h, prune_dim, exclude_dim):
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
        return prune_channels


    def base_unstruct(self, h):
        flattened_h = h.view(h.size(0), -1)
        norm_along_dim_1 = torch.linalg.vector_norm(flattened_h, ord=self.prune_norm, dim=1)
        _, sorted_indices = torch.sort(norm_along_dim_1, dim=1)
        num_indices_to_prune = int(self.prune_hyper * sorted_indices.size(1))

        # Select the indices to prune (lowest norms)
        pruned_indices = sorted_indices[:, :num_indices_to_prune]
        return pruned_indices
    
    def base_struct(self, h, prune_dim):


        return





class WeightPruning(BasePruning):

    def __init__(self):
        BasePruning.__init__(self)
        pass

    def prune_weights(self, weights):
        # Implement weight pruning logic
        # ...
        return pruned_weights

    def apply_pruning_to_layer(self, layer):
        layer_type = self.get_layer_type(layer)
        if hasattr(layer, 'weight'):
            pruned_weights = self.prune_weights(layer.weight.data)
            layer.weight.data = pruned_weights
        return layer


