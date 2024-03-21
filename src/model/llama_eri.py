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
            
            new_module = self._create_new_module(prune_name, target, key)

            self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            print(
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

            # weight = torch.index_select(self.weight, dim=1, index=preserve_channels.to(self.weight.device))
            weight = self.weight[:, preserve_channels.to(self.weight.device)]
            return weight
           
        
    def extract_out_weight(self, input_dim, pruned_dim, preserve_channels, layer_type):
        
        if layer_type == 'linear':
            #  [batch_size, seq_lens, token_lens]
            if input_dim != 3:
                raise ValueError('Not valid input dimension')
                
            # weight = torch.index_select(self.weight, dim=0, index=preserve_channels.to(self.weight.device))
            weight = self.weight[preserve_channels.to(self.weight.device), :]
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
        self.in_features = in_features
        self.prune_metric = cfg['prune_metric']

        self.stream1 = torch.cuda.Stream()
        self.samples_num = torch.zeros((in_features), device=self.weight.data.device)

        self.async_interbatch_weight = None
        self.async_intrabatch_weight = None

        if 'meanglobalinput' in cfg['prune_method']:
            self.mean_for_all_batches = torch.zeros((cfg['seq_len'], in_features), device=self.weight.data.device)
            self.variance_for_all_batches = torch.zeros((cfg['seq_len'], in_features), device=self.weight.data.device)

        self.baseline_inp = torch.zeros((in_features), device=self.weight.data.device)
        if 'wandasp' in self.prune_metric or 'probe' in self.prune_metric:
            self.scaler_inp = torch.zeros((in_features), device=self.weight.data.device)
        elif self.prune_metric == "flap":
            self.fluc_inp = torch.zeros((in_features), device=self.weight.data.device)
        else:
            raise ValueError(f"Unknown pruning method {self.prune_name}")
        self.nsamples = torch.zeros(in_features, dtype=torch.int32, device=self.weight.data.device)
        
        if 'savemetricseq' in cfg['prune_method']:
            self.baseline_inp = torch.zeros((cfg['seq_len'], in_features), device=self.weight.data.device)
            if 'wandasp' in self.prune_metric or 'probe' in self.prune_metric:
                self.scaler_inp = torch.zeros((cfg['seq_len'], in_features), device=self.weight.data.device)
            elif self.prune_metric == "flap":
                self.fluc_inp = torch.zeros((cfg['seq_len'], in_features), device=self.weight.data.device)
            else:
                raise ValueError(f"Unknown pruning method {self.prune_name}")
        self.position_distribution_1 = []
        self.position_distribution_2 = []
        self.position_distribution_3 = []
        self.position_distribution_4 = []
        self.position_distribution_5 = []

    def update_global_input_distribution(self, inp, update_indices):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        # batch_size = inp.shape[0] if batch_size is None else batch_size
        batch_size = inp.shape[0]
        default_batch_size = cfg[cfg['model_name']]['batch_size']['test']
        if batch_size > default_batch_size or inp.shape[0] > default_batch_size:
            raise ValueError(f"Batch size {batch_size} is larger than the default batch size {default_batch_size}")
        
        cur_device = inp.device
        self.mean_for_all_batches = self.mean_for_all_batches.to(cur_device)
        self.variance_for_all_batches = self.variance_for_all_batches.to(cur_device)
        self.samples_num = self.samples_num.to(cur_device)
        update_indices = update_indices.to(cur_device)

        for new_sample in inp:
            # sample_count += 1
            # delta = new_sample - running_mean
            # running_mean += delta / sample_count
            # delta2 = new_sample - running_mean  # Recalculate delta after updating the mean
            # running_variance += delta * delta2
            self.samples_num[update_indices] += 1
            delta = new_sample - self.mean_for_all_batches[:, update_indices]
            self.mean_for_all_batches[:, update_indices] += delta / self.samples_num[update_indices]
            delta2 = new_sample - self.mean_for_all_batches[:, update_indices]  # Recalculate delta after updating the mean
            self.variance_for_all_batches[:, update_indices] += delta * delta2
            # print('new_sample', new_sample.shape, flush=True)
            if 'up_proj' in self.key:
                # print('self.key', self.key, flush=True)
                self.position_distribution_1.append(new_sample[0][0].item())
                self.position_distribution_2.append(new_sample[1][20].item())
                self.position_distribution_3.append(new_sample[2][400].item())
                self.position_distribution_4.append(new_sample[3][600].item())
                self.position_distribution_5.append(new_sample[4][800].item())
        # print('self.position_1', len(self.position_1), flush=True)
        # running_variance_increment = torch.zeros((inp.shape[1]), device=inp.device, dtype=torch.float32)
        # self.mean_for_all_batches = self.mean_for_all_batches.to(cur_device)
        # self.samples_num = self.samples_num.to(cur_device)
        # update_indices = update_indices.to(cur_device)
        # self.mean_for_all_batches[update_indices] *= self.samples_num[update_indices] / (self.samples_num[update_indices] + batch_size)
        # inp = torch.clamp(inp, min=None, max=65504)
        # denominator = (self.samples_num[update_indices] + batch_size)
        # self.mean_for_all_batches[update_indices] += inp / denominator

        # self.variance_for_all_batches[update_indices] +=


    def update_global_metric_score_distribution_ema(self, inp, update_indices, is_probe=False):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        # batch_size = inp.shape[0] if batch_size is None else batch_size
        batch_size = inp.shape[0]
        default_batch_size = cfg[cfg['model_name']]['batch_size']['test']
        if batch_size > default_batch_size or inp.shape[0] > default_batch_size:
            raise ValueError(f"Batch size {batch_size} is larger than the default batch size {default_batch_size}")
        cur_device = inp.device

        momentum = cfg['ema_momentum']
        if is_probe and 'compressfillpbmetric' in cfg['prune_method']:
            momentum = 1 - ((1 - momentum) / (default_batch_size))
        if 'savemetricseq' in cfg['prune_method']:
            if 'wandasp' in self.prune_metric or 'probe' in self.prune_metric:
                self.scaler_inp = self.scaler_inp.to(cur_device)
                self.nsamples = self.nsamples.to(cur_device)
                update_indices = update_indices.to(cur_device)
                self.scaler_inp[:, update_indices] *= momentum
                if cfg['calibration_stage'] == True:
                    norm_squared = torch.clamp(torch.norm(inp, p=2, dim=0) ** 2, min=None, max=65504)
                elif cfg['calibration_stage'] == False:
                    norm_squared = torch.clamp(torch.norm(inp, p=2, dim=0) ** 2, min=None, max=65504)
                # denominator = (self.nsamples[update_indices] + )
                self.scaler_inp[:, update_indices] += (1 - momentum) * norm_squared / batch_size
        else:
            if 'wandasp' in self.prune_metric or 'probe' in self.prune_metric:
                self.scaler_inp = self.scaler_inp.to(cur_device)
                self.nsamples = self.nsamples.to(cur_device)
                update_indices = update_indices.to(cur_device)

                self.scaler_inp[update_indices] *= momentum
                norm_squared = torch.clamp(torch.norm(inp, p=2, dim=(0,1)) ** 2, min=None, max=65504)
                # print(f'{self.key}_norm_squared', norm_squared, flush=True)
                # print('update_indices', update_indices.shape, flush=True)
                # print('norm_squared', norm_squared.shape, flush=True)
                # denominator = (self.nsamples[update_indices] + batch_size)
                # print('self.scaler_inp', self.scaler_inp.shape, flush=True)
                # print('norm_squared', norm_squared.shape, flush=True)
                # print('update_indices', update_indices.shape, flush=True)
                self.scaler_inp[update_indices] += (1 - momentum) * norm_squared / batch_size
                # print('self.scaler_inp', self.scaler_inp, flush=True)
            elif self.prune_metric == "flap":
               pass
        self.nsamples[update_indices] += batch_size

    # def update_global_metric_score_distribution(self, inp, update_indices, batch_size=None, is_probe=False):
    def update_global_metric_score_distribution(self, inp, update_indices):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        # batch_size = inp.shape[0] if batch_size is None else batch_size
        batch_size = inp.shape[0]
        default_batch_size = cfg[cfg['model_name']]['batch_size']['test']
        if batch_size > default_batch_size or inp.shape[0] > default_batch_size:
            raise ValueError(f"Batch size {batch_size} is larger than the default batch size {default_batch_size}")
        cur_device = inp.device

        if 'savemetricseq' in cfg['prune_method']:
            if 'wandasp' in self.prune_metric or 'probe' in self.prune_metric:
                self.scaler_inp = self.scaler_inp.to(cur_device)
                self.nsamples = self.nsamples.to(cur_device)
                update_indices = update_indices.to(cur_device)

                self.scaler_inp[:, update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + batch_size)
                if cfg['calibration_stage'] == True:
                    norm_squared = torch.clamp(torch.norm(inp, p=2, dim=0) ** 2, min=None, max=65504)
                elif cfg['calibration_stage'] == False:
                    # if 'coeff' in cfg['prune_method']:
                    #     # Assuming inp is your input tensor
                    #     # inp_square = torch.clamp(inp ** 2, min=None, max=65504)

                    #     # # Directly compute the sum of squares once and reuse it
                    #     # sum_inp_square = torch.sum(inp_square, dim=0, keepdim=True) + 1e-10
                    #     # sum_inp_square_clamped = torch.clamp(sum_inp_square, min=None, max=65504)

                    #     # # Calculate ratio, notice that clamping sum_inp_square_clamped again is unnecessary
                    #     # # ratio = inp_square / sum_inp_square_clamped
                    #     # ratio = inp_square / inp_square
                    #     # # print('ratio', ratio.shape, flush=True)

                    #     # # Calculate norm_squared without redundant clamping of sum_inp_square
                    #     # norm_squared = torch.clamp(torch.sum(ratio * inp_square, dim=0), min=None, max=65504)

                    #     square_x = torch.clamp(torch.square(inp).to(torch.float32), min=None, max=65504)
                    #     sum_across_bsz = torch.clamp(square_x.sum(dim=0, keepdim=True), min=None, max=65504)
                    #     # proportion = abs_x / torch.sum(abs_x, dim=0, keepdim=True)
                    #     proportion = (square_x / (sum_across_bsz + 1e-10)).to(inp.dtype)
                    #     # Check for NaN values
                    #     has_nan = torch.isnan(proportion).any()

                    #     # Check for infinite values
                    #     has_inf = torch.isinf(proportion).any()

                    #     if has_nan or has_inf:
                    #         print("proportion contains NaN or infinite values.")
                    #         # print(torch.sort(proportion)[0])
                    #         nan_indices = torch.where(torch.isnan(proportion))

                    #         # Print the indices of NaN values
                    #         print("Indices of NaN values:", nan_indices)

                    #         # Print the NaN values using the indices (optional, as they are known to be NaN)
                    #         print("NaN values at these indices:", square_x[nan_indices],  proportion[nan_indices])
                    #     else:
                    #         # print("proportion does not contain NaN or infinite values.")
                    #         pass
                    #     # print('proportion ', proportion, flush=True)
                    #     # proportion = 10
                    #     # print('proportion ', proportion, flush=True)
                    #     norm_squared = torch.sum(square_x * proportion, dim=0)
                    # else:
                    norm_squared = torch.clamp(torch.norm(inp, p=2, dim=0) ** 2, min=None, max=65504)
                # norm_squared = torch.clamp(torch.norm(inp, p=2, dim=0), min=None, max=65504)
                # print(f'{self.key}_norm_squared', norm_squared, flush=True)
                denominator = (self.nsamples[update_indices] + batch_size)
                self.scaler_inp[:, update_indices] += norm_squared / denominator

        else:
            if 'wandasp' in self.prune_metric or 'probe' in self.prune_metric:
                self.scaler_inp = self.scaler_inp.to(cur_device)
                self.nsamples = self.nsamples.to(cur_device)
                update_indices = update_indices.to(cur_device)
                # print('self.key', self.key, flush=True)
                # print('shape', self.scaler_inp.shape)
                # print("self.scaler_inp[update_indices].shape:", self.scaler_inp[update_indices].shape)
                # print("self.nsamples[update_indices].shape:", self.nsamples[update_indices].shape)
                # print("batch_size:", batch_size)
                self.scaler_inp[update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + batch_size)
                norm_squared = torch.clamp(torch.norm(inp, p=2, dim=(0,1)) ** 2, min=None, max=65504)
                # print(f'{self.key}_norm_squared', norm_squared, flush=True)
                # print('update_indices', update_indices.shape, flush=True)
                # print('norm_squared', norm_squared.shape, flush=True)
                denominator = (self.nsamples[update_indices] + batch_size)
                self.scaler_inp[update_indices] += norm_squared / denominator
                # print('self.scaler_inp', self.scaler_inp, flush=True)
            elif self.prune_metric == "flap":
                # TODO: update later
                old_baseline_inp = self.baseline_inp
                self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
                self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
                self.baseline_inp = self.baseline_inp.to(cur_device)
                old_baseline_inp = old_baseline_inp.to(cur_device)
                if self.nsamples == 0:
                    pass
                else:
                    self.fluc_inp = self.fluc_inp.to(cur_device)
                    self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
                    self.fluc_inp += torch.sum((inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1) / (self.nsamples + batch_size)   # a²+b²+c²...没开根号
        self.nsamples[update_indices] += batch_size

    def get_global_input_distribution(self):
        # return mean and std
        # print('self.samples_num', self.samples_num, flush=True)
        return self.mean_for_all_batches, torch.sqrt(self.variance_for_all_batches / (self.samples_num - 1))
        
    def get_global_metric_score_distribution(self):
        if 'wandasp' in self.prune_metric or 'probe' in self.prune_metric:
            return self.scaler_inp
        elif self.prune_metric == "flap":
            return self.fluc_inp


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

    def check_fill_case(self):
        qk_prune_way = cfg['qk_prune_way']
        vo_prune_way = cfg['vo_prune_way']
        if qk_prune_way is None:
            qk_prune_way = vo_prune_way
        if qk_prune_way == 'fill' and ('q_proj' in self.key or 'k_proj' in self.key):
            return True
        elif vo_prune_way == 'fill' and ('v_proj' in self.key or 'o_proj' in self.key):
            return True
        return False

    def prepare_async_interbatch_weight(self, probe_out_dim_indices):

        return
    
    def prepare_async_intrabatch_weight(self):
        return
    
    def prepare_async_weight(self):
        if cfg['async_way'] == None:
            return
        elif cfg['async_way'] == 'interbatch':
            self.prepare_async_interbatch_weight()
        elif cfg['async_way'] == 'intrabatch':
            self.prepare_async_intrabatch_weight()
        return
    
    def get_weight(self):
        if cfg['async_way'] == None:
            return self.weight
        elif cfg['async_way'] == 'interbatch':
            return self.async_interbatch_weight
        elif cfg['async_way'] == 'intrabatch':
            return self.async_intrabatch_weight
    
    
    # no bias in llama-2
    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            forward_start_time = time.time()
            previous_dtype = x.dtype
            weight = self.get_weight()
            if cfg['calibration_stage'] == True:
                # print('calibration_stage', flush=True)
                self.update_global_metric_score_distribution(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                result = F.linear(x, weight, bias=None)
                    # print('calibrateresult', result.dtype, result.shape, result, flush=True)
                result = result.to(previous_dtype)
                return result
                # if 'meanglobalinput' in cfg['prune_method']:
                #     self.update_global_input_distribution(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))

                # metric_score = self.get_global_metric_score_distribution()
                # mean_for_all_batches, std_for_all_batches = self.get_global_input_distribution()
            elif cfg['calibration_stage'] == False:
                if 'probe_in_dim_indices' in kwargs:
                    if 'attn' in self.key:
                        if self.check_fill_case():
                            x = x[..., kwargs['probe_in_dim_indices'].to(weight.device)]

                
                if 'o_proj' in self.key or 'down_proj' in self.key:
                    
                    update_time = time.time()
                    if 'runningmean' in cfg['prune_method']:
                        if 'probe_in_dim_indices' in kwargs:
                            # print('self.key', self.key, flush=True)
                            self.update_global_metric_score_distribution(x, kwargs['probe_in_dim_indices'])
                        else:
                            self.update_global_metric_score_distribution(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                    elif 'ema' in cfg['prune_method']:
                        if 'probe_in_dim_indices' in kwargs:
                            self.update_global_metric_score_distribution_ema(x, kwargs['probe_in_dim_indices'])
                        else:
                            self.update_global_metric_score_distribution_ema(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                    update_time_end = time.time()
                    print('update_time', update_time_end - update_time, flush=True)

                if 'probe' in cfg['prune_method'] and 'cal_mlp_probe_out_dim_metric' in kwargs and kwargs['cal_mlp_probe_out_dim_metric'] == True:
                    # print('probeinput', x.dtype, x.shape, x, flush=True)
                    # print('probeweight', self.weight.dtype, self.weight.shape, self.weight, flush=True)
                    # x = x.to(torch.float32)
                    # weight = self.weight.to(torch.float32)
                    # print('probeinput 2 ', x.dtype, x.shape, x, flush=True)
                    # for i in range(x.shape[-2]):
                    #     print('\n\n', i)
                    #     for j in range(x.shape[-1]):
                    #         print(x[:, i, j])
                    # print('probeweight2 ', weight.dtype, weight.shape, weight, flush=True)
                    # if 'square' in cfg['prune_method']:
                    #     result = torch.clamp(F.linear(x, self.weight ** 2, bias=None), min=None, max=65504)
                    #     print('probesquareresult', result.dtype, result.shape, result, flush=True)
                    # else:
                    result = F.linear(x, weight, bias=None)
                    # print('proberesult', result.dtype, result.shape, result, flush=True)
                    result = result.to(previous_dtype)
                    return result
                elif 'probe' in cfg['prune_method'] and 'cal_attn_probe_out_dim_metric' in kwargs and kwargs['cal_attn_probe_out_dim_metric'] == True:                 
                    result = F.linear(x, weight, bias=None)
                    result = result.to(previous_dtype)
                    return result

                input_dim = x.dim()
                batch_size = x.size(0)
                seq_len = x.size(1)

                linear_layer_info = {
                    'weight': weight.data,
                }

                if 'probe_out_dim_indices' in kwargs:
                    if 'attn' in self.key:
                        weight = weight[kwargs['probe_out_dim_indices'].to(weight.device), :]
                        if self.check_fill_case():
                            # print('fill', flush=True)
                            self.out_selected_dim = kwargs['probe_out_dim_indices']
                    else:
                        weight = weight[kwargs['probe_out_dim_indices'].to(weight.device), :]
                    result = F.linear(x, weight, bias=None)
                    if 'attn' in self.key:
                        if self.check_fill_case():
                            # print('fill', flush=True)
                            refilling_output = torch.zeros(batch_size, seq_len, self.out_features, device=result.device, dtype=result.dtype)
                            refilling_output[..., self.out_selected_dim] = result
                            result = refilling_output
                    result = result.to(previous_dtype)
                    return result
                elif 'probe_in_dim_indices' in kwargs:
                    if 'attn' in self.key:
                        weight = weight[:, kwargs['probe_in_dim_indices'].to(weight.device)]
                    else:
                        weight = weight[:, kwargs['probe_in_dim_indices'].to(weight.device)]
                    result = F.linear(x, weight, bias=None)
                    result = result.to(previous_dtype)
                    return result
                else:
                    # only prune input in each layer
                    if 'traditional' in cfg['prune_method']:
                        x, pruned_dim, preserve_channels = self.pruning_module.batch_pruning(x, self.layer_type, linear_layer_info, self.key, self.is_prune_out_dim)
                        weight = self.extract_in_weight(input_dim, pruned_dim, preserve_channels, self.layer_type)
                    else:
                        weight = weight
                    
                if 'flap' in cfg['prune_metric']:
                    # all_channels = torch.arange(self.in_features, dtype=preserve_channels.dtype, device=preserve_channels.device)
                    # mask = all_channels[preserve_channels]
                    # mean_x = torch.mean(x, dim=1)
                    # flap = (x - mean_x) * 
                    # output_bias = ((mean_inp * ~mask.to(self.weight.data.device)) @ self.weight.data.T)
                    # result = F.linear(x, weight, bias=output_bias)
                    pass
                else:
                    # print('calibrateinput', x.dtype, x.shape, x, flush=True)
                    # print('calibrateweight', weight.dtype, weight.shape, weight, flush=True)
                    # for i in range(x.shape[-2]):
                    #     print(x[:, i, :])
                    # x = x.to(torch.float32)
                    # weight = weight.to(torch.float32)
                    # print('calibrateinput 2', x.dtype, x.shape, x, flush=True)
                    # print('calibrateweight 2', weight.dtype, weight.shape, weight, flush=True)
                    # print('here')
                    result = F.linear(x, weight, bias=None)
                    # print('calibrateresult', result.dtype, result.shape, result, flush=True)
                result = result.to(previous_dtype)
                forward_end_time = time.time()
                print('forward_time', forward_end_time - forward_start_time, flush=True)





            forward_start_time = time.time()
            previous_dtype = x.dtype
            weight = self.get_weight()
            if cfg['calibration_stage'] == True:
                # print('calibration_stage', flush=True)
                self.update_global_metric_score_distribution(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                result = F.linear(x, weight, bias=None)
                    # print('calibrateresult', result.dtype, result.shape, result, flush=True)
                result = result.to(previous_dtype)
                return result
                # if 'meanglobalinput' in cfg['prune_method']:
                #     self.update_global_input_distribution(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))

                # metric_score = self.get_global_metric_score_distribution()
                # mean_for_all_batches, std_for_all_batches = self.get_global_input_distribution()
            elif cfg['calibration_stage'] == False:
                if 'probe_in_dim_indices' in kwargs:
                    if 'attn' in self.key:
                        if self.check_fill_case():
                            x = x[..., kwargs['probe_in_dim_indices'].to(weight.device)]

                
                if 'o_proj' in self.key or 'down_proj' in self.key:
                    with torch.cuda.stream(self.stream1):
                        update_time = time.time()
                        if 'runningmean' in cfg['prune_method']:
                            if 'probe_in_dim_indices' in kwargs:
                                # print('self.key', self.key, flush=True)
                                self.update_global_metric_score_distribution(x, kwargs['probe_in_dim_indices'])
                            else:
                                self.update_global_metric_score_distribution(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                        elif 'ema' in cfg['prune_method']:
                            if 'probe_in_dim_indices' in kwargs:
                                self.update_global_metric_score_distribution_ema(x, kwargs['probe_in_dim_indices'])
                            else:
                                self.update_global_metric_score_distribution_ema(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                        update_time_end = time.time()
                        print('update_time1', update_time_end - update_time, flush=True)

                if 'probe' in cfg['prune_method'] and 'cal_mlp_probe_out_dim_metric' in kwargs and kwargs['cal_mlp_probe_out_dim_metric'] == True:
                    # print('probeinput', x.dtype, x.shape, x, flush=True)
                    # print('probeweight', self.weight.dtype, self.weight.shape, self.weight, flush=True)
                    # x = x.to(torch.float32)
                    # weight = self.weight.to(torch.float32)
                    # print('probeinput 2 ', x.dtype, x.shape, x, flush=True)
                    # for i in range(x.shape[-2]):
                    #     print('\n\n', i)
                    #     for j in range(x.shape[-1]):
                    #         print(x[:, i, j])
                    # print('probeweight2 ', weight.dtype, weight.shape, weight, flush=True)
                    # if 'square' in cfg['prune_method']:
                    #     result = torch.clamp(F.linear(x, self.weight ** 2, bias=None), min=None, max=65504)
                    #     print('probesquareresult', result.dtype, result.shape, result, flush=True)
                    # else:
                    result = F.linear(x, weight, bias=None)
                    # print('proberesult', result.dtype, result.shape, result, flush=True)
                    result = result.to(previous_dtype)
                    return result
                elif 'probe' in cfg['prune_method'] and 'cal_attn_probe_out_dim_metric' in kwargs and kwargs['cal_attn_probe_out_dim_metric'] == True:                 
                    result = F.linear(x, weight, bias=None)
                    result = result.to(previous_dtype)
                    return result

                input_dim = x.dim()
                batch_size = x.size(0)
                seq_len = x.size(1)

                linear_layer_info = {
                    'weight': weight.data,
                }

                if 'probe_out_dim_indices' in kwargs:
                    if 'attn' in self.key:
                        weight = weight[kwargs['probe_out_dim_indices'].to(weight.device), :]
                        if self.check_fill_case():
                            # print('fill', flush=True)
                            self.out_selected_dim = kwargs['probe_out_dim_indices']
                    else:
                        weight = weight[kwargs['probe_out_dim_indices'].to(weight.device), :]
                    result = F.linear(x, weight, bias=None)
                    if 'attn' in self.key:
                        if self.check_fill_case():
                            # print('fill', flush=True)
                            refilling_output = torch.zeros(batch_size, seq_len, self.out_features, device=result.device, dtype=result.dtype)
                            refilling_output[..., self.out_selected_dim] = result
                            result = refilling_output
                    result = result.to(previous_dtype)
                    return result
                elif 'probe_in_dim_indices' in kwargs:
                    if 'attn' in self.key:
                        weight = weight[:, kwargs['probe_in_dim_indices'].to(weight.device)]
                    else:
                        weight = weight[:, kwargs['probe_in_dim_indices'].to(weight.device)]
                    result = F.linear(x, weight, bias=None)
                    result = result.to(previous_dtype)
                    return result
                else:
                    # only prune input in each layer
                    if 'traditional' in cfg['prune_method']:
                        x, pruned_dim, preserve_channels = self.pruning_module.batch_pruning(x, self.layer_type, linear_layer_info, self.key, self.is_prune_out_dim)
                        weight = self.extract_in_weight(input_dim, pruned_dim, preserve_channels, self.layer_type)
                    else:
                        weight = weight
                    
                if 'flap' in cfg['prune_metric']:
                    # all_channels = torch.arange(self.in_features, dtype=preserve_channels.dtype, device=preserve_channels.device)
                    # mask = all_channels[preserve_channels]
                    # mean_x = torch.mean(x, dim=1)
                    # flap = (x - mean_x) * 
                    # output_bias = ((mean_inp * ~mask.to(self.weight.data.device)) @ self.weight.data.T)
                    # result = F.linear(x, weight, bias=output_bias)
                    pass
                else:
                    # print('calibrateinput', x.dtype, x.shape, x, flush=True)
                    # print('calibrateweight', weight.dtype, weight.shape, weight, flush=True)
                    # for i in range(x.shape[-2]):
                    #     print(x[:, i, :])
                    # x = x.to(torch.float32)
                    # weight = weight.to(torch.float32)
                    # print('calibrateinput 2', x.dtype, x.shape, x, flush=True)
                    # print('calibrateweight 2', weight.dtype, weight.shape, weight, flush=True)
                    # print('here')
                    result = F.linear(x, weight, bias=None)
                    # print('calibrateresult', result.dtype, result.shape, result, flush=True)
                result = result.to(previous_dtype)
                forward_end_time = time.time()
                print('forward_time1', forward_end_time - forward_start_time, flush=True)
                return result