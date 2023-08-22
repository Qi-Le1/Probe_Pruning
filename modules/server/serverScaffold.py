from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import models
from itertools import compress
from config import cfg
from collections import defaultdict

from .serverBase import ClientSampler

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

from optimizer.api import create_optimizer
from .serverBase import ServerBase

from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)

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

class ServerScaffold(ServerBase):

    def __init__(
        self, 
        model: ModelType,
        clients: dict[int, ClientType],
        dataset: DatasetType
    ) -> None:

        super().__init__(dataset=dataset)
        self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        server_optimizer = create_optimizer(model, 'server')
        self.server_optimizer_state_dict = server_optimizer.state_dict()
        self.clients = clients
        self.c_global = copy.deepcopy(model)
        self.c_global.apply(init_param)
    
    def train(
        self,
        dataset: DatasetType,  
        optimizer: OptimizerType, 
        metric: MetricType, 
        logger: LoggerType, 
        global_epoch: int
    ):
        # print(f'global_epoch is: {global_epoch}')
        logger.safe(True)
        selected_client_ids, num_active_clients = super().select_clients(clients=self.clients)
        super().distribute_server_model_to_clients(
            server_model_state_dict=self.server_model_state_dict,
            clients=self.clients
        )
        start_time = time.time()
        lr = optimizer.param_groups[0]['lr']

        data_loader_list = []
        for client_id in selected_client_ids:
            client_sampler = ClientSampler(
                batch_size=cfg['client']['batch_size']['train'], 
                data_split=self.clients[client_id].data_split['train'],
                max_local_gradient_update=cfg['local_epoch']*len(self.clients[client_id].data_split['train']),
                client_id=client_id
            )
            data_loader_list.append(make_data_loader(
                dataset={'train': dataset}, 
                tag='client',
                batch_sampler={'train': client_sampler}
            )['train']) 

        total_delta = copy.deepcopy(self.server_model_state_dict)
        total_delta = to_device(total_delta, cfg['device'])
        for key in total_delta:
            total_delta[key] = 0.0

        for i in range(num_active_clients):
            m = selected_client_ids[i]
            self.clients[m].active = True
            c_delta_para = self.clients[m].train(
                # dataset=dataset_m, 
                data_loader=data_loader_list[i],
                lr=lr, 
                metric=metric, 
                logger=logger,
                grad_updates_num=cfg['max_local_gradient_update'],
                c_global=self.c_global
            )

            for key in total_delta:
                total_delta[key] += c_delta_para[key]

            super().add_log(
                i=i,
                num_active_clients=num_active_clients,
                start_time=start_time,
                global_epoch=global_epoch,
                lr=lr,
                selected_client_ids=selected_client_ids,
                metric=metric,
                logger=logger,
            )
            
        for key in total_delta:
            total_delta[key] /= cfg['num_clients']
        c_global_para = self.c_global.state_dict()
        for key in c_global_para:
            c_global_para[key] += total_delta[key]
        self.c_global.load_state_dict(c_global_para)

        logger.safe(False)
        logger.reset()
        super().update_server_model(clients=self.clients) 
        return
    
    def evaluate_trained_model(
        self,
        dataset,
        batchnorm_dataset,
        logger,
        metric,
        global_epoch
    ):  
        return super().evaluate_trained_model(
            dataset=dataset,
            batchnorm_dataset=batchnorm_dataset,
            logger=logger,
            metric=metric,
            global_epoch=global_epoch,
            server_model_state_dict=self.server_model_state_dict
        )