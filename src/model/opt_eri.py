import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from module import check_skip_layers


class OPTEriModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.add_pruner()

    def add_pruner(self):
        self._find_and_replace()
        mark_no_trainable(self.model)        
        return
    
    def _create_new_module(self, target, key):
        bias = hasattr(target, "bias") and target.bias is not None

        in_features = getattr(target, 'in_features', None)
        out_features = getattr(target, 'out_features', None)
        
        kwargs = {
            "prune_metric": cfg['prune_metric'],
            "key": key,
            "dev": target.weight.device,
        }
        
        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` is supported."
            )
        new_module = Linear(in_features, out_features, bias=bias, **kwargs)

        return new_module

    def _find_and_replace(self):
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        # return
        target_modules = _get_target_modules(cfg)
        print('target_modules: ', target_modules)
        for key in key_list:
            print('key', key)
            if 'dense' in cfg['prune_method'] or 'llmpruner' in cfg['prune_method'] or 'loraprune' in cfg['prune_method']:
                continue

            if not _check_target_module_exists(target_modules, key):
                continue
            
            if check_skip_layers(key):
                continue

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
        new_module.weight = old_module.weight
        new_module.weight.requires_grad = False
        new_module.device = old_module.weight.device
        new_module.is_pruned = True
        if hasattr(old_module, "bias"):
            new_module.bias = old_module.bias

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


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
        # print('key', self.key, weight.shape)
        return weight[indices.to(self.weight.device), :]
        # return torch.index_select(weight, dim=0, index=indices.to(self.weight.device))
    
    def extract_bias(self, bias, indices):
        return bias[indices.to(self.bias.device)]
        
class Linear(nn.Linear, EriLayer):
    def __init__(
        self,
        in_features,
        out_features,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=False)
        EriLayer.__init__(self, in_features=in_features, out_features=out_features, **kwargs)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.layer_type = 'linear'
        self.in_features = in_features
        self.prune_metric = cfg['prune_metric']

        self.async_interbatch_weight = None
        self.async_intrabatch_weight = None

        self.async_interbatch_bias = None
        self.async_intrabatch_bias = None

        self.async_interbatch_in_dim_indices = None
        self.async_intrabatch_out_dim_indices = None
        self.async_intrabatch_in_dim_indices = None

        self.retrieve_weight = torch.cuda.Event(enable_timing=False, blocking=False)

        self.compensate_bias = None

        if ('out_proj' in self.key or 'fc2' in self.key):
            self.nsamples = torch.zeros(in_features, dtype=torch.int32, device=self.weight.data.device)   
            if 'wandasp' in self.prune_metric:
                self.scaler_inp = torch.zeros((cfg['max_seq_len'], in_features), device=self.weight.data.device, dtype=cfg['data_type'])
                if 'bias' in cfg['prune_method']:
                    self.baseline_inp = torch.zeros((cfg['max_seq_len'], in_features), device=self.weight.data.device, dtype=cfg['data_type'])
            elif "flap" in self.prune_metric:
                self.fluc_inp = torch.zeros((cfg['max_seq_len'], in_features), device=self.weight.data.device, dtype=cfg['data_type'])
                self.baseline_inp = torch.zeros((cfg['max_seq_len'], in_features), device=self.weight.data.device, dtype=cfg['data_type'])
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
            self.scaler_inp[:seq_len, update_indices] *= momentum

            if 'bias' in cfg['prune_method']:
                self.baseline_inp = self.baseline_inp.to(cur_device)
                self.baseline_inp[:seq_len, update_indices] *= momentum
                self.baseline_inp[:seq_len, update_indices] += (1 - momentum) * (torch.mean(inp, dim=0) / batch_size)

            if cfg['calibration_stage'] == True:
                norm_squared = torch.linalg.vector_norm(inp, ord=2, dim=0) ** 2
                self.scaler_inp[:seq_len, update_indices] += (1 - momentum) * norm_squared / batch_size
            elif cfg['calibration_stage'] == False:
                if cfg['pad_tokens'] is not None:
                    cfg['nonpad_tokens_denominator'] = cfg['nonpad_tokens_denominator'].to(cur_device)
                    norm_squared = torch.clamp(torch.linalg.vector_norm(inp, ord=2, dim=0) ** 2, max=cfg['data_type_max'])
                    self.scaler_inp[:seq_len, update_indices] += (1 - momentum) * torch.clamp(norm_squared / cfg['nonpad_tokens_denominator'], max=cfg['data_type_max'])
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
                # flaps github code is not matching the paper formula: https://github.com/CASIA-IVA-Lab/FLAP
                # If bsz is 1, it is not variance between the average of current batch channel and the average of channel.
                # It is the sum of the variance between each element in the channel and the average of the channel.
                # We follow their github code
                self.fluc_inp[:seq_len, update_indices] *= (self.nsamples[update_indices] - 1) / (self.nsamples[update_indices] + batch_size - 1)
                self.fluc_inp[:seq_len, update_indices] += torch.sum((inp - torch.mean(self.baseline_inp[:seq_len, update_indices], dim=0).unsqueeze(0).unsqueeze(0)) * (inp - torch.mean(old_baseline_inp[:seq_len, update_indices], dim=0).unsqueeze(0).unsqueeze(0)), dim=0) / (self.nsamples[update_indices] + batch_size) 

    def update_global_metric_score_distribution(self, inp, update_indices):
        if cfg['cur_batch_index'] == 0:
            return
        
        if len(inp.shape) == 2:
            raise ValueError(f"Input shape {inp.shape} is not supported. Please provide a 3D tensor.")
        

        batch_size = inp.shape[0]
        seq_len = inp.shape[1]
        cur_device = inp.device
        update_indices = update_indices.to(cur_device)
        self.nsamples = self.nsamples.to(cur_device)
        
        if 'wandasp' in self.prune_metric:
            
            self.scaler_inp = self.scaler_inp.to(cur_device)
            if 'bias' in cfg['prune_method']:
                self.baseline_inp = self.baseline_inp.to(cur_device)
                self.baseline_inp[:seq_len, update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + batch_size)
                self.baseline_inp[:seq_len, update_indices] += torch.mean(inp, dim=0) / (self.nsamples[update_indices] + batch_size)
                        
            if cfg['calibration_stage'] == True:
                self.scaler_inp[:seq_len, update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + batch_size)
                norm_squared = torch.linalg.vector_norm(inp, ord=2, dim=0) ** 2
                denominator = (self.nsamples[update_indices] + batch_size)
                self.scaler_inp[:seq_len, update_indices] += norm_squared / denominator
            elif cfg['calibration_stage'] == False:
                self.scaler_inp = self.scaler_inp.to(cfg['data_type'])
                if cfg['pad_tokens'] is not None:
                    cfg['nonpad_tokens_denominator'] = cfg['nonpad_tokens_denominator'].to(cur_device)
                    self.scaler_inp[:seq_len, update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + cfg['nonpad_tokens_denominator'])
                    norm_squared = torch.clamp(torch.linalg.vector_norm(inp, ord=2, dim=0) ** 2, max=cfg['data_type_max'])
                    denominator = (self.nsamples[update_indices] + cfg['nonpad_tokens_denominator'])
                    self.scaler_inp[:seq_len, update_indices] += torch.clamp(norm_squared / denominator, max=cfg['data_type_max'])
                else:
                    self.scaler_inp[:seq_len, update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + batch_size)
                    norm_squared = torch.clamp(torch.linalg.vector_norm(inp, ord=2, dim=0) ** 2, max=cfg['data_type_max'])
                    denominator = (self.nsamples[update_indices] + batch_size)
                    self.scaler_inp[:seq_len, update_indices] += torch.clamp(norm_squared / denominator, max=cfg['data_type_max'])


        elif "flap" in self.prune_metric:
            self.baseline_inp = self.baseline_inp.to(cur_device)
            self.fluc_inp = self.fluc_inp.to(cur_device)
            
            old_baseline_inp = self.baseline_inp.clone()
            self.baseline_inp[:seq_len, update_indices] *= self.nsamples[update_indices] / (self.nsamples[update_indices] + batch_size)
            self.baseline_inp[:seq_len, update_indices] += torch.mean(inp, dim=0) / (self.nsamples[update_indices] + batch_size)
            
            if torch.all(self.nsamples == 0):
                pass
            else:
                self.fluc_inp[:seq_len, update_indices] *= (self.nsamples[update_indices] - 1) / (self.nsamples[update_indices] + batch_size - 1)
                # flaps github code is not matching the paper formula: https://github.com/CASIA-IVA-Lab/FLAP
                # If bsz is 1, it is not variance between the average of current batch channel and the average of channel.
                # It is the sum of the variance between each element in the channel and the average of the channel.
                # We follow their github code
                self.fluc_inp[:seq_len, update_indices] += torch.sum((inp - torch.mean(self.baseline_inp[:seq_len, update_indices], dim=0).unsqueeze(0).unsqueeze(0)) * (inp - torch.mean(old_baseline_inp[:seq_len, update_indices], dim=0).unsqueeze(0).unsqueeze(0)), dim=0) / (self.nsamples[update_indices] + batch_size)  
        if cfg['pad_tokens'] is not None:
            self.nsamples[update_indices] += cfg['nonpad_tokens_denominator']
        else:
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

    def return_global_metric_info(self):
        if ('out_proj' in self.key or 'fc2' in self.key):
            if 'wandasp' in self.prune_metric:
                if 'bias' in cfg['prune_method']:
                   return {
                        'nsamples': self.nsamples,
                        'baseline_inp': self.baseline_inp,
                        'scaler_inp': self.scaler_inp
                    } 
                else:
                    return {
                        'nsamples': self.nsamples,
                        'scaler_inp': self.scaler_inp
                    }
            elif "flap" in self.prune_metric:
                return {
                    'nsamples': self.nsamples,
                    'baseline_inp': self.baseline_inp,
                    'fluc_inp': self.fluc_inp
                }
            else:
                raise ValueError(f"Unknown pruning metric")
        else:
            return None

    def set_global_metric_to_data_type(self):
        if ('out_proj' in self.key or 'fc2' in self.key):
            if 'wandasp' in self.prune_metric:
                if 'bias' in cfg['prune_method']:
                    self.baseline_inp = self.baseline_inp.to(cfg['data_type'])
                    self.scaler_inp = self.scaler_inp.to(cfg['data_type'])
                else:
                    self.scaler_inp = self.scaler_inp.to(cfg['data_type'])
            elif "flap" in self.prune_metric:
                self.baseline_inp = self.baseline_inp.to(cfg['data_type'])
                self.fluc_inp = self.fluc_inp.to(cfg['data_type'])
            else:
                raise ValueError(f"Unknown pruning metric")



    def prepare_async_interbatch_weight(self, **kwargs):
        if 'out_dim_indices' in kwargs:
            self.async_interbatch_bias = self.bias[kwargs['out_dim_indices']]
            self.async_interbatch_weight = self.extract_out_dim_weight(self.weight, kwargs['out_dim_indices'])
        elif 'in_dim_indices' in kwargs:
            print('biasshape', self.key, self.bias.shape)
            self.async_interbatch_bias = self.bias
            self.async_interbatch_weight = self.extract_in_dim_weight(self.weight, kwargs['in_dim_indices'])
            # record indices to update metric
            self.async_interbatch_in_dim_indices = kwargs['in_dim_indices']
        else:
            raise ValueError('Not valid input')
        return
    
    def prepare_async_intrabatch_weight(self, **kwargs):
        if 'out_dim_indices' in kwargs:
            self.async_intrabatch_out_dim_indices = kwargs['out_dim_indices']
        elif 'in_dim_indices' in kwargs:
            self.async_intrabatch_in_dim_indices = kwargs['in_dim_indices']
        else:
            self.async_intrabatch_in_dim_indices = torch.arange(self.in_features, dtype=torch.int).to(device=self.weight.device)

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
            device = self.weight.device
            stream = torch.cuda.current_stream(device=device)
            stream.wait_event(self.retrieve_weight)    
            return self.async_interbatch_weight
        elif cfg['mode'] == 'asyncintra':    
            return self.weight
        
    def get_bias(self):        
        if cfg['mode'] == 'sync':
            return self.bias
        elif cfg['mode'] == 'asyncinter':
            if cfg['cur_batch_index'] == 0:
                return self.bias
            return self.async_interbatch_bias
        elif cfg['mode'] == 'asyncintra':    
            return self.bias

    
    def get_async_in_dim_indices(self):
        if cfg['mode'] == 'asyncinter':
            return self.async_interbatch_in_dim_indices
        elif cfg['mode'] == 'asyncintra':
            return self.async_intrabatch_in_dim_indices
    
    def get_compensate_bias(self, x, weight, in_dim_indices):
        # return torch.zeros(weight.shape[0], device=x.device)
        if cfg['cur_batch_index'] == 0:
            return torch.zeros(weight.shape[0], device=x.device)

        if cfg['mode'] == 'asyncinter':
            if self.compensate_bias == None:
                calib = torch.mean(self.baseline_inp, dim=0)
                calib = calib.to(x.device)
                in_dim_indices = in_dim_indices.to(device=x.device)
                calib[in_dim_indices] = 0
                compensate_bias = F.linear(calib, weight, bias=None)
            else:
                compensate_bias = self.compensate_bias
        else:
            calib = torch.mean(self.baseline_inp, dim=0)
            calib = calib.to(x.device)
            in_dim_indices = in_dim_indices.to(device=x.device)
            calib[in_dim_indices] = 0
            compensate_bias = F.linear(calib, weight, bias=None)
        return compensate_bias
    
    # opt has bias
    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            forward_start_time = time.time()
            previous_dtype = x.dtype
            if cfg['calibration_stage'] == True:
                if 'out_proj' in self.key or 'fc2' in self.key:
                    self.update_global_metric_score_distribution(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                result = F.linear(x, self.weight, bias=self.bias)
                result = result.to(previous_dtype)
                return result
            elif cfg['calibration_stage'] == False:
                if cfg['cur_batch_index'] == 0:
                    self.set_global_metric_to_data_type()
                    
                if 'probe' in cfg['prune_method'] and 'cal_mlp_probe_out_dim_metric' in kwargs and kwargs['cal_mlp_probe_out_dim_metric'] == True:
                    result = F.linear(x, self.weight, bias=self.bias)
                    result = result.to(previous_dtype)
                    return result
                elif 'probe' in cfg['prune_method'] and 'cal_attn_probe_out_dim_metric' in kwargs and kwargs['cal_attn_probe_out_dim_metric'] == True:               
                    result = F.linear(x, self.weight, bias=self.bias)
                    result = result.to(previous_dtype)
                    return result
        
                if cfg['mode'] == 'asyncinter':
                    weight = self.get_weight()
                    bias = self.get_bias()
                    if 'out_proj' in self.key or 'fc2' in self.key:
                        async_in_dim_indices = self.get_async_in_dim_indices()
                        if 'runningmean' in cfg['prune_method']:
                            self.update_global_metric_score_distribution(x, async_in_dim_indices)    
                        elif 'ema' in cfg['prune_method']:
                            self.update_global_metric_score_distribution_ema(x, async_in_dim_indices)
                    previous_dtype = x.dtype
                    result = F.linear(x, weight, bias=bias)

                    if 'out_proj' in self.key or 'fc2' in self.key:
                        if 'bias' in cfg['prune_method']:
                            compensate_bias = self.get_compensate_bias(x, self.weight, async_in_dim_indices)
                            result += compensate_bias
                    result = result.to(previous_dtype)
                    return result
                elif cfg['mode'] == 'asyncintra':
                    weight = self.get_weight()
                    bias = self.get_bias()
                    if 'out_proj' in self.key or 'fc2' in self.key:
                        async_in_dim_indices = self.get_async_in_dim_indices()
                        if 'runningmean' in cfg['prune_method']:
                            self.update_global_metric_score_distribution(x, async_in_dim_indices)    
                        elif 'ema' in cfg['prune_method']:
                            self.update_global_metric_score_distribution_ema(x, async_in_dim_indices)
       
                    if 'out_dim_indices' in kwargs:
                        weight = self.extract_out_dim_weight(weight, self.async_intrabatch_out_dim_indices)
                        bias = self.extract_bias(bias, self.async_intrabatch_out_dim_indices)
                        result = F.linear(x, weight, bias=bias)
                        result = result.to(previous_dtype)
                        return result
                    elif 'in_dim_indices' in kwargs:
                        weight = self.extract_in_dim_weight(weight, self.async_intrabatch_in_dim_indices)
                        result = F.linear(x, weight, bias=bias)
                        if 'out_proj' in self.key or 'fc2' in self.key:
                            if 'bias' in cfg['prune_method']:
                                compensate_bias = self.get_compensate_bias(x, self.weight, self.async_intrabatch_in_dim_indices)
                                result += compensate_bias
                        result = result.to(previous_dtype)
                        return result
                    else:
                        result = F.linear(x, weight, bias=None)
                    return result
                elif cfg['mode'] == 'sync':
                    weight = self.get_weight()
                    bias = self.get_bias()
                    if 'out_proj' in self.key or 'fc2' in self.key:
                        if 'runningmean' in cfg['prune_method']:
                            if 'in_dim_indices' in kwargs:
                                self.update_global_metric_score_distribution(x, kwargs['in_dim_indices'])
                            else:
                                self.update_global_metric_score_distribution(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                        elif 'ema' in cfg['prune_method']:
                            if 'in_dim_indices' in kwargs:
                                self.update_global_metric_score_distribution_ema(x, kwargs['in_dim_indices'])
                            else:
                                self.update_global_metric_score_distribution_ema(x, torch.arange(self.in_features, dtype=torch.long).to(device=x.device))
                    
                    if 'out_dim_indices' in kwargs:
                        weight = self.extract_out_dim_weight(weight, kwargs['out_dim_indices'])
                        bias = self.extract_bias(bias, kwargs['out_dim_indices'])
                        result = F.linear(x, weight, bias=bias)
                        result = result.to(previous_dtype)
                        return result
                    elif 'in_dim_indices' in kwargs:
                        weight = self.extract_in_dim_weight(weight, kwargs['in_dim_indices'])
                        result = F.linear(x, weight, bias=bias)
                        if 'out_proj' in self.key or 'fc2' in self.key:
                            if 'bias' in cfg['prune_method']:
                                compensate_bias = self.get_compensate_bias(x, self.weight, async_in_dim_indices)
                                result += compensate_bias
                        result = result.to(previous_dtype)
                        return result
                    else:
                        result = F.linear(x, weight, bias=bias)
                    
                    result = result.to(previous_dtype)
                return result