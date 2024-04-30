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


class LlamaEriModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.add_pruner()

    def add_pruner(self):
        self._find_and_replace()
        mark_no_trainable(self.model)        
        return
    
    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

    def _create_new_module(self, target, key):
        bias = hasattr(target, "bias") and target.bias is not None
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        in_features = getattr(target, 'in_features', None)
        out_features = getattr(target, 'out_features', None)
        
        kwargs = {
            "prune_metric": cfg['prune_metric'],
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
            new_module = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, **kwargs)
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
            new_module = Linear(in_features, out_features, bias=bias, **kwargs)

        return new_module

    def _find_and_replace(self):
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key, module in self.model.named_modules():
            print('key: ', key, type(module), flush=True)
        # return
        target_modules = _get_target_modules(cfg)
        print('target_modules: ', target_modules)
        for key in key_list:
            if 'dense' in cfg['prune_method']:
                continue

            if not _check_target_module_exists(target_modules, key):
                continue
            
            layer_order_matches = re.findall(r'\d+', key)
            if layer_order_matches:  # Check if the list is not empty
                layer_order = int(layer_order_matches[0])  # Convert the first match to an integer
                if layer_order <= cfg['skip_layers']:
                    continue
            else:
                raise ValueError(f"Layer order not found in the layer key {key}")

            print('Replaced Layer Keys', key, flush=True)

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            
            new_module = self._create_new_module(target, key)
            
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

def _check_target_module_exists(target_modules, key):
    if isinstance(target_modules, str):
        target_module_found = re.fullmatch(target_modules, key)
    else:
        target_module_found = any(key.endswith(target_key) for target_key in target_modules)
    return target_module_found

class EriLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.key = kwargs['key']
        return
        
    def extract_in_dim_weight(self, weight, indices):
        return weight[:, indices.to(self.weight.device)]
        # return torch.index_select(weight, dim=1, index=indices.to(self.weight.device))
           
        
    def extract_out_dim_weight(self, weight, indices):
        return weight[indices.to(self.weight.device), :]
        # return torch.index_select(weight, dim=0, index=indices.to(self.weight.device))
        
class Linear(nn.Linear, EriLayer):
    def __init__(
        self,
        in_features,
        out_features,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=False)
        EriLayer.__init__(self, in_features=in_features, out_features=out_features, **kwargs)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.layer_type = 'linear'
        self.in_features = in_features
        self.prune_metric = cfg['prune_metric']

        self.async_interbatch_weight = None
        self.async_intrabatch_weight = None

        self.async_interbatch_weight_index = torch.tensor(-1).to(self.weight.data.device)
        self.async_intrabatch_weight_index = torch.tensor(-1).to(self.weight.data.device)

        if ('o_proj' in self.key or 'down_proj' in self.key) and cfg['mode'] == 'asyncinter':
            self.async_interbatch_in_dim_indices = torch.arange((in_features), device=self.weight.data.device, dtype=torch.int32)

        self.nsamples = torch.zeros(in_features, dtype=torch.int32, device=self.weight.data.device)
        # set same shape for all baselines for comparison
        self.baseline_inp = torch.zeros((cfg['max_seq_len'], in_features), device=self.weight.data.device, dtype=cfg['data_type'])
        if 'wandasp' in self.prune_metric:
            self.scaler_inp = torch.zeros((cfg['max_seq_len'], in_features), device=self.weight.data.device, dtype=cfg['data_type'])
        elif "flap" in self.prune_metric:
            self.fluc_inp = torch.zeros((cfg['max_seq_len'], in_features), device=self.weight.data.device, dtype=cfg['data_type'])
        else:
            raise ValueError(f"Unknown pruning method")


    def update_global_metric_score_distribution_ema(self, inp, update_indices):
        if cfg['cur_batch_index'] == 0:
            return
        
        if len(inp.shape) == 2:
            raise ValueError(f"Input shape {inp.shape} is not supported. Please provide a 3D tensor.")
 
        batch_size = inp.shape[0]
        seq_len = inp.shape[1]
        momentum = cfg['ema_momentum']
        cur_device = inp.device
        update_indices = update_indices.to(cur_device)
        self.nsamples = self.nsamples.to(cur_device)

        if 'wandasp' in self.prune_metric:
            self.scaler_inp = self.scaler_inp.to(cur_device)
            self.scaler_inp[:, update_indices] *= momentum

            if 'bias' in cfg['prune_method']:
                self.baseline_inp = self.baseline_inp.to(cur_device)
                self.baseline_inp[:seq_len, update_indices] *= momentum
                self.baseline_inp[:seq_len, update_indices] += (1 - momentum) * (torch.mean(inp, dim=0) / batch_size)

            if cfg['calibration_stage'] == True:
                norm_squared = torch.clamp(torch.linalg.vector_norm(inp, ord=2, dim=0) ** 2, max=cfg['data_type_max'])
                self.scaler_inp[:seq_len, update_indices] += (1 - momentum) * torch.clamp(norm_squared / batch_size, max=cfg['data_type_max'])
            elif cfg['calibration_stage'] == False:
                if cfg['pad_tokens'] is not None:
                    cfg['pad_tokens'] = cfg['pad_tokens'].to(cur_device)
                    cfg['pad_tokens_denominator'] = cfg['pad_tokens_denominator'].to(cur_device)
                    inp[cfg['pad_tokens']] = 0
                    norm_squared = torch.clamp(torch.linalg.vector_norm(inp, ord=2, dim=0) ** 2, max=cfg['data_type_max'])
                    # self.scaler_inp[:seq_len, update_indices] += (1 - momentum) * torch.clamp(norm_squared / cfg['pad_tokens_denominator'], max=cfg['data_type_max'])
                    self.scaler_inp[:seq_len, update_indices] += (1 - momentum) * torch.clamp(norm_squared / batch_size, max=cfg['data_type_max'])
                else:
                    norm_squared = torch.clamp(torch.linalg.vector_norm(inp, ord=2, dim=0) ** 2, max=cfg['data_type_max'])
                    self.scaler_inp[:seq_len, update_indices] += (1 - momentum) * torch.clamp(norm_squared / batch_size, max=cfg['data_type_max'])
        elif "flap" in self.prune_metric:
            self.baseline_inp = self.baseline_inp.to(cur_device)
            self.fluc_inp = self.fluc_inp.to(cur_device)

            old_baseline_inp = self.baseline_inp.clone()
            self.baseline_inp[:seq_len, update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + batch_size)
            self.baseline_inp[:seq_len, update_indices] += torch.mean(inp, dim=0) / (self.nsamples[update_indices] + batch_size)
            
            if torch.all(self.nsamples == 0):
                pass
            else:
                self.fluc_inp[:, update_indices] *= (self.nsamples[update_indices] - 1) / (self.nsamples[update_indices] + batch_size - 1)
                self.fluc_inp[:, update_indices] += torch.sum((inp - torch.mean(self.baseline_inp[:, update_indices], dim=0).unsqueeze(0).unsqueeze(0)) * (inp - torch.mean(old_baseline_inp[:, update_indices], dim=0).unsqueeze(0).unsqueeze(0)), dim=0) / (self.nsamples[update_indices] + batch_size) 

    def update_global_metric_score_distribution(self, inp, update_indices):
        if cfg['cur_batch_index'] == 0:
            return
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        default_batch_size = cfg[cfg['model_name']]['batch_size']['test']
        if batch_size > default_batch_size or inp.shape[0] > default_batch_size:
            raise ValueError(f"Batch size {batch_size} is larger than the default batch size {default_batch_size}")
        cur_device = inp.device
        update_indices = update_indices.to(cur_device)
        self.nsamples = self.nsamples.to(cur_device)

        if 'wandasp' in self.prune_metric:
            self.scaler_inp = self.scaler_inp.to(cur_device)

            self.scaler_inp[:, update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + batch_size)

            if 'bias' in cfg['prune_method']:
                self.baseline_inp = self.baseline_inp.to(cur_device)
                self.baseline_inp[:, update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + batch_size)
                self.baseline_inp[:, update_indices] += torch.mean(inp, dim=0) / (self.nsamples[update_indices] + batch_size)
                        
            if cfg['calibration_stage'] == True:
                norm_squared = torch.clamp(torch.linalg.vector_norm(inp, ord=2, dim=0) ** 2, max=cfg['data_type_max'])
            elif cfg['calibration_stage'] == False:
                norm_squared = torch.clamp(torch.linalg.vector_norm(inp, ord=2, dim=0) ** 2, max=cfg['data_type_max'])
            denominator = (self.nsamples[update_indices] + batch_size)
            self.scaler_inp[:, update_indices] += torch.clamp(norm_squared / denominator, max=cfg['data_type_max'])
        elif "flap" in self.prune_metric:
            self.baseline_inp = self.baseline_inp.to(cur_device)
            self.fluc_inp = self.fluc_inp.to(cur_device)

            old_baseline_inp = self.baseline_inp.clone()
            self.baseline_inp[:, update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + batch_size)
            self.baseline_inp[:, update_indices] += torch.mean(inp, dim=0) / (self.nsamples[update_indices] + batch_size)
            
            if torch.all(self.nsamples == 0):
                pass
            else:
                self.fluc_inp[:, update_indices] *= (self.nsamples[update_indices] - 1) / (self.nsamples[update_indices] + batch_size - 1)
                # flaps github code is not matching the paper formula: https://github.com/CASIA-IVA-Lab/FLAP
                # If bsz is 1, it is not variance between the average of current batch channel and the average of channel.
                # It is the sum of the variance between each element in the channel and the average of the channel.
                # We follow their github code
                self.fluc_inp[:, update_indices] += torch.sum((inp - torch.mean(self.baseline_inp[:, update_indices], dim=0).unsqueeze(0).unsqueeze(0)) * (inp - torch.mean(old_baseline_inp[:, update_indices], dim=0).unsqueeze(0).unsqueeze(0)), dim=0) / (self.nsamples[update_indices] + batch_size)  
        self.nsamples[update_indices] += batch_size

        
    def get_global_metric_score_distribution(self, cur_seq_len=None):
        if 'wandasp' in self.prune_metric:
            return self.scaler_inp if cur_seq_len is None else self.scaler_inp[:cur_seq_len]
        elif "flap" in self.prune_metric:
            return self.fluc_inp if cur_seq_len is None else self.fluc_inp[:cur_seq_len]
        else:
            raise ValueError(f"Unknown pruning metric")

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


    def prepare_async_interbatch_weight(self, **kwargs):
        if 'out_dim_indices' in kwargs:
            self.async_interbatch_weight = self.extract_out_dim_weight(self.weight, kwargs['out_dim_indices'])
        elif 'in_dim_indices' in kwargs:
            self.async_interbatch_weight = self.extract_in_dim_weight(self.weight, kwargs['in_dim_indices'])
            # record indices to update metric
            self.async_interbatch_in_dim_indices = kwargs['in_dim_indices']
        else:
            raise ValueError('Not valid input')
        if self.async_interbatch_weight_index.device != self.weight.device:
            self.async_interbatch_weight_index = self.async_interbatch_weight_index.to(self.weight.device)
        self.async_interbatch_weight_index += 1
        return
    
    def prepare_async_intrabatch_weight(self, **kwargs):
        if 'out_dim_indices' in kwargs:
            self.async_intrabatch_weight = self.extract_out_dim_weight(self.weight, kwargs['out_dim_indices'])
        elif 'in_dim_indices' in kwargs:
            self.async_intrabatch_weight = self.extract_in_dim_weight(self.weight, kwargs['in_dim_indices'])
            # record indices to update metric
            self.async_intrabatch_in_dim_indices = kwargs['in_dim_indices']
        else:
            raise ValueError('Not valid input')
        # print('herererere')
        if self.async_intrabatch_weight_index.device != self.weight.device:
            self.async_intrabatch_weight_index = self.async_intrabatch_weight_index.to(self.weight.device)
        self.async_intrabatch_weight_index += 1
        return

    
    def prepare_async_weight(self, **kwargs):
        if cfg['mode'] == 'sync':
            pass
        elif cfg['mode'] == 'asyncinter':
            self.prepare_async_interbatch_weight(**kwargs)
        elif cfg['mode'] == 'asyncintra':
            self.prepare_async_intrabatch_weight(**kwargs)
        else:
            raise ValueError('Not valid mode')
        return
    
    def get_weight(self):        
        if cfg['mode'] == 'sync':
            return self.weight
        elif cfg['mode'] == 'asyncinter':
            if cfg['cur_batch_index'] == 0:
                return self.weight
            # dont have the prepared weight for current batch
            # sync the stream1
            if 'ema' in cfg['prune_method'] or 'runningmean' in cfg['prune_method']:
                # if self.async_interbatch_weight_index.item() != cfg['cur_batch_index'] - 1:
                #     torch.cuda.synchronize(cfg['cuda_stream1'])
                #     # time.sleep(0.001)
                #     print('wait sync weight step end', self.key, self.async_interbatch_weight_index)
                
                while self.async_interbatch_weight_index.item() != cfg['cur_batch_index'] - 1:
                    # torch.cuda.synchronize(cfg['cuda_stream1'])
                    time.sleep(0.001)
                    print('wait sync weight step end', self.key, self.async_interbatch_weight_index)
            return self.async_interbatch_weight
        elif cfg['mode'] == 'asyncintra':
            # if self.async_intrabatch_weight_index.item() != cfg['cur_batch_index']:
            #     # time.sleep(0.001)  # Sleep for 1 millisecond
            # # if self.async_intrabatch_weight_index != cfg['cur_batch_index']:
            #     torch.cuda.synchronize(cfg['cuda_stream1'])
            #     print('wait sync weight step end', self.key, self.async_intrabatch_weight_index)
            
            while self.async_intrabatch_weight_index.item() != cfg['cur_batch_index']:
                time.sleep(0.001)  # Sleep for 1 millisecond
            # if self.async_intrabatch_weight_index != cfg['cur_batch_index']:
                # torch.cuda.synchronize(cfg['cuda_stream1'])
                print('wait sync weight step end', self.key, self.async_intrabatch_weight_index)
            return self.async_intrabatch_weight
    
    def get_async_in_dim_indices(self):
        if cfg['mode'] == 'asyncinter':
            return self.async_interbatch_in_dim_indices
        elif cfg['mode'] == 'asyncintra':
            return self.async_intrabatch_in_dim_indices
    
    # no bias in llama-2
    def forward(self, x: torch.Tensor, **kwargs):
        # print('forward key', self.key, flush=True)
        with torch.no_grad():
            forward_start_time = time.time()
            previous_dtype = x.dtype
            if cfg['calibration_stage'] == True:
                # running mean
                self.update_global_metric_score_distribution(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                result = F.linear(x, self.weight, bias=None)
                    # print('calibrateresult', result.dtype, result.shape, result, flush=True)
                result = result.to(previous_dtype)
                return result
                # if 'meanglobalinput' in cfg['prune_method']:
                #     self.update_global_input_distribution(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))

                # metric_score = self.get_global_metric_score_distribution()
                # mean_for_all_batches, std_for_all_batches = self.get_global_input_distribution()
            elif cfg['calibration_stage'] == False:
                
                if 'probe' in cfg['prune_method'] and 'cal_mlp_probe_out_dim_metric' in kwargs and kwargs['cal_mlp_probe_out_dim_metric'] == True:
                    result = F.linear(x, self.weight, bias=None)
                    result = result.to(previous_dtype)
                    return result
                elif 'probe' in cfg['prune_method'] and 'cal_attn_probe_out_dim_metric' in kwargs and kwargs['cal_attn_probe_out_dim_metric'] == True:               
                    result = F.linear(x, self.weight, bias=None)
                    result = result.to(previous_dtype)
                    return result
        
                if cfg['mode'] == 'asyncinter':
                    weight = self.get_weight()
                    # print('weight', weight.dtype, weight.shape, self.key, flush=True)
                    if 'o_proj' in self.key or 'down_proj' in self.key:
                        # torch.cuda.synchronize(cfg['cuda_default_stream'])
                        update_time = time.time()
                        # with torch.cuda.stream(cfg['cuda_stream1']):
                        update_time = time.time()
                        async_in_dim_indices = self.get_async_in_dim_indices()


                        if 'runningmean' in cfg['prune_method']:
                            self.update_global_metric_score_distribution(x, async_in_dim_indices)    
                        elif 'ema' in cfg['prune_method']:
                            self.update_global_metric_score_distribution_ema(x, async_in_dim_indices)
                        # torch.cuda.synchronize(cfg['cuda_default_stream'])
                        update_time_end = time.time()
                        # print('metric update_time', update_time_end - update_time, flush=True)
                        # print('update_time', update_time_end - update_time, flush=True)
                        # torch.cuda.synchronize(cfg['cuda_stream1'])
                    previous_dtype = x.dtype
                    result = F.linear(x, weight, bias=None)

                    if 'o_proj' in self.key or 'down_proj' in self.key:
                        # 'flap' in cfg['prune_metric'] and 
                        if 'bias' in cfg['prune_method']:
                            calib = torch.mean(self.baseline_inp, dim=0)
                            calib = calib.to(x.device)
                            # all_indices = torch.arange(self.in_features, dtype=torch.long).to(device=x.device)
                            async_in_dim_indices = async_in_dim_indices.to(device=x.device)
                            # print('all_indices', all_indices.dtype, all_indices.shape, all_indices.device, flush=True)
                            print('async_in_dim_indices', async_in_dim_indices.dtype, async_in_dim_indices.shape, async_in_dim_indices.device, flush=True)
                            calib[async_in_dim_indices] = 0
                            # pruned_channel_indices = all_indices[mask]
                            compensate_bias = F.linear(calib, self.weight, bias=None)
                            result += compensate_bias
                    result = result.to(previous_dtype)
                    return result
                elif cfg['mode'] == 'asyncintra':
                    weight = self.get_weight()
                    # print('weight', weight.dtype, weight.shape, self.key, flush=True)
                    if 'o_proj' in self.key or 'down_proj' in self.key:
                        # torch.cuda.synchronize(cfg['cuda_default_stream'])
                        update_time = time.time()
                        # with torch.cuda.stream(cfg['cuda_stream1']):
                        update_time = time.time()
                        async_in_dim_indices = self.get_async_in_dim_indices()
                        if 'runningmean' in cfg['prune_method']:
                            self.update_global_metric_score_distribution(x, async_in_dim_indices)    
                        elif 'ema' in cfg['prune_method']:
                            self.update_global_metric_score_distribution_ema(x, async_in_dim_indices)
                        # torch.cuda.synchronize(cfg['cuda_default_stream'])
                        update_time_end = time.time()
                        # print('metric update_time', update_time_end - update_time, flush=True)
                        # print('update_time', update_time_end - update_time, flush=True)
                        # torch.cuda.synchronize(cfg['cuda_stream1'])
                    previous_dtype = x.dtype
                    result = F.linear(x, weight, bias=None)

                    if 'o_proj' in self.key or 'down_proj' in self.key:
                        if 'flap' in cfg['prune_metric']:
                            calib = torch.mean(self.baseline_inp, dim=0)
                            calib = calib.to(x.device)
                            # all_indices = torch.arange(self.in_features, dtype=torch.long).to(device=x.device)
                            async_in_dim_indices = async_in_dim_indices.to(device=x.device)
                            # print('all_indices', all_indices.dtype, all_indices.shape, all_indices.device, flush=True)
                            print('async_in_dim_indices', async_in_dim_indices.dtype, async_in_dim_indices.shape, async_in_dim_indices.device, flush=True)
                            calib[async_in_dim_indices] = 0
                            # pruned_channel_indices = all_indices[mask]
                            compensate_bias = F.linear(calib, self.weight, bias=None)
                            result += compensate_bias
                    result = result.to(previous_dtype)
                    return result
                elif cfg['mode'] == 'sync':
                    weight = self.get_weight()
                    if 'o_proj' in self.key or 'down_proj' in self.key:
                        update_time = time.time()
                        if 'runningmean' in cfg['prune_method']:
                            if 'in_dim_indices' in kwargs:
                                # print('self.key', self.key, flush=True)
                                self.update_global_metric_score_distribution(x, kwargs['in_dim_indices'])
                            else:
                                self.update_global_metric_score_distribution(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                        elif 'ema' in cfg['prune_method']:
                            if 'in_dim_indices' in kwargs:
                                # print("kwargs['in_dim_indices']", kwargs['in_dim_indices'])
                                self.update_global_metric_score_distribution_ema(x, kwargs['in_dim_indices'])
                            else:
                                self.update_global_metric_score_distribution_ema(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                        update_time_end = time.time()
                        if cfg['logger_detailed_info'] == True:
                            print('metric update_time', update_time_end - update_time, flush=True)

                    

                    input_dim = x.dim()
                    batch_size = x.size(0)
                    seq_len = x.size(1)

                    linear_layer_info = {
                        'weight': weight.data,
                    }

                    if 'out_dim_indices' in kwargs:
                        if 'attn' in self.key:
                            # weight = weight[kwargs['out_dim_indices'].to(weight.device), :]
                            # weight = torch.index_select(weight, dim=0, index=kwargs['out_dim_indices'].to(weight.device))
                            weight = self.extract_out_dim_weight(weight, kwargs['out_dim_indices'])
                            # print('attn weight', weight.shape, flush=True)
                            # if self.check_fill_case():
                            #     # print('fill', flush=True)
                            #     self.out_selected_dim = kwargs['out_dim_indices']
                        else:
                            # weight = weight[kwargs['out_dim_indices'].to(weight.device), :]
                            # weight = torch.index_select(weight, dim=0, index=kwargs['out_dim_indices'].to(weight.device))
                            weight = self.extract_out_dim_weight(weight, kwargs['out_dim_indices'])

                        result = F.linear(x, weight, bias=None)
                        # if 'attn' in self.key:
                        #     if self.check_fill_case():
                        #         # print('fill', flush=True)
                        #         refilling_output = torch.zeros(batch_size, seq_len, self.out_features, device=result.device, dtype=result.dtype)
                        #         refilling_output[..., self.out_selected_dim] = result
                        #         result = refilling_output
                        result = result.to(previous_dtype)
                        return result
                    elif 'in_dim_indices' in kwargs:
                        if 'attn' in self.key:
                            # weight = weight[:, kwargs['in_dim_indices'].to(weight.device)]
                            # weight = torch.index_select(weight, dim=1, index=kwargs['in_dim_indices'].to(weight.device))
                            weight = self.extract_in_dim_weight(weight, kwargs['in_dim_indices'])
                        else:
                            # weight = weight[:, kwargs['in_dim_indices'].to(weight.device)]
                            # weight = torch.index_select(weight, dim=1, index=kwargs['in_dim_indices'].to(weight.device))
                            weight = self.extract_in_dim_weight(weight, kwargs['in_dim_indices'])

                        result = F.linear(x, weight, bias=None)
                        if 'o_proj' in self.key or 'down_proj' in self.key:
                            # 'flap' in cfg['prune_metric'] and 
                            if 'bias' in cfg['prune_method']:
                                calib = torch.mean(self.baseline_inp, dim=0)
                                calib = calib.to(x.device)
                                # all_indices = torch.arange(self.in_features, dtype=torch.long).to(device=x.device)
                                in_dim_indices = kwargs['in_dim_indices'].to(device=x.device)
                                # print('all_indices', all_indices.dtype, all_indices.shape, all_indices.device, flush=True)
                                print('in_dim_indices', in_dim_indices.dtype, in_dim_indices.shape, in_dim_indices.device, flush=True)
                                calib[in_dim_indices] = 0
                                # pruned_channel_indices = all_indices[mask]
                                compensate_bias = F.linear(calib, self.weight, bias=None)
                                result += compensate_bias
                        result = result.to(previous_dtype)
                        return result
                    else:
                        result = F.linear(x, weight, bias=None)
                    
                    result = result.to(previous_dtype)
                    forward_end_time = time.time()
                    print('forward_time', forward_end_time - forward_start_time, flush=True)

                return result