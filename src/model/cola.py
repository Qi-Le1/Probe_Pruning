import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from config import cfg
from .model import init_param, mse_loss
from module import to_device
from module.peft.tuners.cola import ColaLayer

warnings.filterwarnings('ignore', category=ConvergenceWarning)


class LowRank(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.input_size = model_cfg['input_size']
        self.output_size = model_cfg['output_size']
        self.hidden_size = model_cfg['hidden_size']
        self.mode = model_cfg['mode']
        if model_cfg['dropout'] > 0.0:
            self.dropout = nn.Dropout(p=model_cfg['dropout'])
        else:
            self.dropout = nn.Identity()
        if self.mode == 'linear':
            self.cola_A = nn.Linear(self.input_size, self.hidden_size, bias=False)
            self.cola_B = nn.Linear(self.hidden_size, self.output_size, bias=False)
        elif self.mode == 'conv2d':
            self.kernel_size = model_cfg['kernel_size']
            self.stride = model_cfg['stride']
            self.padding = model_cfg['padding']
            self.cola_A = nn.Conv2d(self.input_size, self.hidden_size, self.kernel_size, self.stride, self.padding,
                                    bias=False)
            self.cola_B = nn.Conv2d(self.hidden_size, self.output_size, (1, 1), (1, 1), bias=False)
        else:
            raise ValueError('Not valid mode')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.cola_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.cola_B.weight)
        return

    def fit(self, input, optimizer, scheduler):
        self.train(True)
        input = to_device(input, cfg['device_cola'])
        output = {}
        x = input['data'].to(self.cola_A.weight.dtype)
        output['target'] = self.forward(x)
        with torch.no_grad():
            input['target'] = (output['target'] - input['target']).detach()
        output['loss'] = 0.5 * F.mse_loss(output['target'], input['target'], reduction='sum')
        output['loss'].backward()
        if cfg['task_name'] in ['ic']:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return output

    def make_delta_weight(self):
        if self.mode == 'linear':
            delta_weight = self.cola_B.weight.data @ self.cola_A.weight.data
        elif self.mode == 'conv2d':
            if self.kernel_size == (1, 1):
                # conv2d 1x1
                delta_weight = (self.cola_B.weight.data.squeeze(3).squeeze(2) @
                                self.cola_A.weight.data.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            else:
                # conv2d 3x3
                delta_weight = F.conv2d(self.cola_A.weight.data.permute(1, 0, 2, 3),
                                        self.cola_B.weight.data).permute(1, 0, 2, 3)
        else:
            raise ValueError('Not valid mode')
        return delta_weight

    def forward(self, x):
        x = self.dropout(x)
        x = self.cola_A(x)
        x = self.cola_B(x)
        return x


class Linear(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.input_size = model_cfg['input_size']
        self.output_size = model_cfg['output_size']
        self.bias = model_cfg['bias']
        self.mode = model_cfg['mode']
        if self.mode == 'linear':
            self.linear = nn.Linear(self.input_size, self.output_size, bias=self.bias)
        elif self.mode == 'conv2d':
            self.kernel_size = model_cfg['kernel_size']
            self.stride = model_cfg['stride']
            self.padding = model_cfg['padding']
            self.linear = nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding,
                                    bias=self.bias)
        else:
            raise ValueError('Not valid mode')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.linear.weight)
        if self.bias:
            nn.init.zeros_(self.linear.bias)
        return

    def fit(self, input, optimizer, scheduler):
        self.train(True)
        input = to_device(input, cfg['device_cola'])
        output = {}
        x = input['data'].to(self.linear.weight.dtype)
        output['target'] = self.forward(x)
        with torch.no_grad():
            input['target'] = (output['target'] - input['target']).detach()
        output['loss'] = 0.5 * F.mse_loss(output['target'], input['target'], reduction='sum')
        output['loss'].backward()
        if cfg['task_name'] in ['ic']:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return output

    def make_delta_weight(self):
        if self.mode == 'linear':
            delta_weight = self.linear.weight.data
        elif self.mode == 'conv2d':
            delta_weight = self.linear.weight.data
        else:
            raise ValueError('Not valid mode')
        if self.bias:
            delta_bias = self.linear.bias.data
            return delta_weight, delta_bias
        return delta_weight

    def forward(self, x):
        x = self.linear(x)
        return x


class MLP(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.input_size = model_cfg['input_size']
        self.output_size = model_cfg['output_size']
        self.hidden_size = model_cfg['hidden_size']
        self.scale_factor = model_cfg['scale_factor']
        self.num_layers = model_cfg['num_layers']
        self.activation = model_cfg['activation']
        self.mode = model_cfg['mode']
        input_size = self.input_size
        hidden_size = self.hidden_size
        blocks = []
        for _ in range(self.num_layers):
            blocks.append(nn.Linear(input_size, hidden_size))
            if self.activation == 'relu':
                blocks.append(nn.ReLU())
            elif self.activation == 'sigmoid':
                blocks.append(nn.Sigmoid())
            else:
                raise ValueError('Not valid activation')
            input_size = hidden_size
            hidden_size = int(hidden_size * self.scale_factor)
        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Linear(input_size, self.output_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        return

    def fit(self, input, optimizer, scheduler):
        self.train(True)
        input = to_device(input, cfg['device_cola'])
        output = {}
        x = input['data'].to(self.linear.weight.dtype)
        output['target'] = self.forward(x)
        with torch.no_grad():
            input['target'] = (output['target'] - input['target']).detach()
        output['loss'] = 0.5 * F.mse_loss(output['target'], input['target'], reduction='sum')
        output['loss'].backward()
        if cfg['task_name'] in ['ic']:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return output

    def make_delta_weight(self):
        raise NotImplementedError

    def forward(self, x):
        if self.mode == 'conv2d':
            x = x.permute(0, 2, 3, 1)
        x = self.blocks(x)
        x = self.linear(x)
        if self.mode == 'conv2d':
            x = x.permute(0, 3, 1, 2)
        return x


class Embedding(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.input_size = model_cfg['input_size']
        self.output_size = model_cfg['output_size']
        self.hidden_size = model_cfg['hidden_size']
        self.padding_idx = model_cfg['padding_idx']
        self.max_norm = model_cfg['max_norm']
        self.norm_type = model_cfg['norm_type']
        self.scale_grad_by_freq = model_cfg['scale_grad_by_freq']
        self.sparse = model_cfg['sparse']
        self.mode = model_cfg['mode']
        if model_cfg['dropout'] > 0.0:
            self.dropout = nn.Dropout(p=model_cfg['dropout'])
        else:
            self.dropout = nn.Identity()
        weight_A = torch.randn((self.hidden_size, self.input_size))
        weight_B = torch.randn((self.output_size, self.hidden_size))
        self.cola_embedding_A = nn.Parameter(weight_A)
        self.cola_embedding_B = nn.Parameter(weight_B)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.cola_embedding_A)
        nn.init.normal_(self.cola_embedding_B)
        return

    def fit(self, input, optimizer, scheduler):
        self.train(True)
        input = to_device(input, cfg['device_cola'])
        output = {}
        x = input['data']
        output['target'] = self.forward(x)
        with torch.no_grad():
            input['target'] = (output['target'] - input['target']).detach()
        output['loss'] = 0.5 * F.mse_loss(output['target'], input['target'], reduction='sum')
        output['loss'].backward()
        if cfg['task_name'] in ['ic']:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return output

    def make_delta_weight(self):
        delta_weight = (self.cola_embedding_B @ self.cola_embedding_A).T
        return delta_weight

    def forward(self, x):
        x = self.dropout(x)
        x = F.embedding(
            x,
            self.cola_embedding_A[self.active_adapter].T,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        x = x @ self.cola_embedding_B.T
        return x


class Router(nn.Module):
    def __init__(self, model, dist_mode):
        super().__init__()
        self.model = nn.ModuleList(model)
        self.dist_mode = dist_mode
        self.split = None
        self.unique_split = None
        self.indices = None
        self.sorted_indices = None

    def make_split(self, split):
        self.split = split
        self.unique_split = torch.unique(split)
        indices = []
        for unique_value in self.unique_split:
            mask_i = self.split == unique_value
            indices_i = torch.nonzero(mask_i).view(-1)
            indices.append(indices_i)
        self.indices = indices
        self.sorted_indices = torch.argsort(torch.cat(indices))
        return

    def fit(self, input, optimizer=None, scheduler=None):
        for i in range(len(self.unique_split)):
            data_i, target_i = input['data'][self.indices[i]], input['target'][self.indices[i]]
            input_i = {'data': data_i, 'target': target_i}
            self.model[self.unique_split[i]].fit(input_i, optimizer[self.unique_split[i]],
                                                 scheduler[self.unique_split[i]])
        return

    def make_delta_weight(self):
        delta_weight = []
        for i in range(len(self.model)):
            delta_weight_i = self.model[i].make_delta_weight()
            delta_weight.append(delta_weight_i)
        delta_weight = torch.stack(delta_weight, dim=-1).mean(dim=-1)
        return delta_weight

    def forward(self, x):
        x_ = []
        for i in range(len(self.unique_split)):
            x_i = x[self.indices[i]]
            x_i = self.model[self.unique_split[i]](x_i)
            x_.append(x_i)
        x_ = torch.cat(x_, dim=0)
        x = x_[self.sorted_indices]
        return x


def make_cola_model(name, model_name, model_cfg):
    if 'classifier' in name and cfg['task_name'] == 'sc':
        model_name = 'linear'
    if model_name == 'lowrank':
        model_cfg['hidden_size'] = cfg['cola']['lowrank']['hidden_size']
        model_cfg['dropout'] = cfg['cola']['lowrank']['dropout']
        model = LowRank(model_cfg)
    elif model_name == 'linear':
        if 'classifier' in name and cfg['task_name'] == 'sc':
            model_cfg['bias'] = True
        elif cfg['task_name'] == 'ic':
            model_cfg['bias'] = True
        else:
            model_cfg['bias'] = cfg['cola']['linear']['bias']
        model = Linear(model_cfg)
    elif model_name == 'mlp':
        model_cfg['hidden_size'] = cfg['cola']['mlp']['hidden_size']
        model_cfg['scale_factor'] = cfg['cola']['mlp']['scale_factor']
        model_cfg['num_layers'] = cfg['cola']['mlp']['num_layers']
        model_cfg['activation'] = cfg['cola']['mlp']['activation']
        model = MLP(model_cfg)
    elif model_name == 'embedding':
        model_cfg['hidden_size'] = cfg['cola']['lowrank']['hidden_size']
        model_cfg['dropout'] = cfg['cola']['lowrank']['dropout']
        model = Embedding(model_cfg)
    else:
        raise ValueError('Not valid model name')
    return model


def make_model_name(model_name):
    model_name_list = model_name.split("~")
    num_model_each = cfg['num_split'] // len(model_name_list)
    remainder = cfg['num_split'] % len(model_name_list)
    model_name_ = []
    for model_name_i in model_name_list:
        model_name_.extend([model_name_i] * num_model_each)
    for i in range(remainder):
        model_name_.append(model_name_list[i])
    return model_name_


def make_cola(model, model_name, dist_mode='joint'):
    if dist_mode in ['alone', 'col']:
        cfg['cola']['model_name'] = make_model_name(model_name)
    cola = {}
    for name, module in model.base_model.named_modules():
        if 'original_module' not in name and isinstance(module, ColaLayer):
            if isinstance(module, nn.Linear):
                input_size = module.in_features
                output_size = module.out_features
                model_cfg = {'input_size': input_size, 'output_size': output_size, 'mode': 'linear'}
            elif isinstance(module, nn.Conv2d):
                input_size = module.in_features
                output_size = module.out_features
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                model_cfg = {'input_size': input_size, 'output_size': output_size, 'kernel_size': kernel_size,
                             'stride': stride, 'padding': padding, 'mode': 'conv2d'}
            elif isinstance(module, nn.Embedding):
                input_size = module.in_features
                output_size = module.out_features
                padding_idx = module.padding_idx
                max_norm = module.max_norm
                norm_type = module.norm_type
                scale_grad_by_freq = module.scale_grad_by_freq
                sparse = module.sparse
                model_cfg = {'input_size': input_size, 'output_size': output_size,
                             'padding_idx': padding_idx, 'max_norm': max_norm,
                             'norm_type': norm_type, 'scale_grad_by_freq': scale_grad_by_freq,
                             'sparse': sparse, 'mode': 'embedding'}
            else:
                raise ValueError('Not valid module')
            if dist_mode in ['alone', 'col']:
                cola[name] = []
                for i in range(cfg['num_split']):
                    cola_model_i = make_cola_model(name, cfg['cola']['model_name'][i], model_cfg)
                    cola[name].append(cola_model_i)
                cola[name] = Router(cola[name], dist_mode)
            else:
                cola[name] = make_cola_model(name, model_name, model_cfg)
    return cola
