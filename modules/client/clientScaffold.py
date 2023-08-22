from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg

# from torchstat import stat

from utils.api import (
    to_device,  
    collate
)

from _typing import (
    DatasetType,
    OptimizerType,
    DataLoaderType,
    ModelType,
    MetricType,
    LoggerType,
    ClientType,
    ServerType
)

from models.api import (
    create_model
)

from optimizer.api import create_optimizer

from data import make_data_loader

from .clientBase import ClientBase

def init_param(m):
    if cfg['model_name'] == 'resnet18':
        if isinstance(m, nn.Conv2d):
            m.weight.data.zero_()
            if m.bias:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
    else:
        if isinstance(m, nn.Conv2d):
            m.weight.data.zero_()
            # if m.bias:
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
    return m

class ClientScaffold(ClientBase):

    def __init__(
        self, 
        client_id: int, 
        model: ModelType, 
        data_split: list[int],
    ) -> None:

        super().__init__()
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self.prev_model_state_dict = copy.deepcopy(self.model_state_dict)
        optimizer = create_optimizer(model, 'client')
        self.optimizer_state_dict = optimizer.state_dict()
        self.active = False
        # self.c_global = None
        self.c_local = copy.deepcopy(model)
        self.c_local.apply(init_param)
        # self.c_local_state_dict = {k: v.cpu() for k, v in self.c_local.state_dict().items()}


    @classmethod
    def create_clients(
        cls,
        model: ModelType, 
        data_split: dict[str, dict[int, list[int]]],
    ) -> dict[int, object]:
        '''
        Create clients which organized in dict type
        
        Parameters
        ----------
        model: ModelType
        data_split: dict[str, dict[int, list[int]]]

        Returns
        -------
        dict[int, object]
        '''
        client_id = torch.arange(cfg['num_clients'])
        clients = [None for _ in range(cfg['num_clients'])]
        for m in range(len(clients)):
            clients[m] = ClientScaffold(
                client_id=client_id[m], 
                model=model, 
                data_split={
                    'train': data_split['train'][m], 
                    'test': data_split['test'][m]
                },
            )
        return clients


    def train(
        self, 
        # dataset: DatasetType, 
        data_loader,
        lr: int, 
        metric: MetricType, 
        logger: LoggerType,
        grad_updates_num: int,
        c_global
    ) -> None:
        
        if grad_updates_num == 0:
            return

        model = create_model(track_running_stats=False, on_cpu=False)
        global_model_para = copy.deepcopy(self.model_state_dict)
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = create_optimizer(model, 'client')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)


        cur_grad_updates_num = 0

        c_global_para = c_global.state_dict()
        c_local_para = self.c_local.state_dict()

        for i, input in enumerate(data_loader):

            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            optimizer.zero_grad()
            output = model(input)
            output['loss'].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['max_clip_norm'])
            optimizer.step()

            net_para = copy.deepcopy(model.state_dict())
            for key in net_para:
                net_para[key] = net_para[key] - lr * (c_global_para[key] - c_local_para[key])
            model.load_state_dict(net_para)

            evaluation = metric.evaluate(
                metric.metric_name['train'], 
                input, 
                output
            )
            logger.append(
                evaluation, 
                'train', 
                n=input_size
            )

            cur_grad_updates_num += 1           

        
        c_new_para = self.c_local.state_dict()
        c_delta_para = copy.deepcopy(self.c_local.state_dict())
        global_model_para = to_device(global_model_para, cfg['device'])
        net_para = model.state_dict()
        for key in net_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cur_grad_updates_num * lr)
            c_delta_para[key] = c_new_para[key] - c_local_para[key]
        self.c_local.load_state_dict(c_new_para)

        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        return c_delta_para
