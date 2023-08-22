from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import models
import collections
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


class ClientFedDyn(ClientBase):

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

        # initiate self.delta_l_k
        self.delta_l_k = copy.deepcopy(list(model.parameters()))
        for param_index, param in enumerate(model.parameters()):
            self.delta_l_k[param_index] = copy.deepcopy(param.data.new_zeros(param.size()))

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
            clients[m] = ClientFedDyn(
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
        feddyn_alpha
    ) -> None:

        model = create_model(track_running_stats=False, on_cpu=False)
        model.load_state_dict(self.model_state_dict, strict=False)

        global_weight_collector = list(model.parameters())
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = create_optimizer(model, 'client')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        feddyn_alpha = to_device(torch.tensor(feddyn_alpha), cfg['device'])

        
        for i, input in enumerate(data_loader):

            input = collate(input)
            # print(f"input[id]: {input['id']}\n")
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            optimizer.zero_grad()
            output = model(input)
            
            local_weight_collector = list(model.parameters())
            # calculate 2 proximal terms
            delta_l_k_model_inner_product = 0.0
            prox_square = 0.0
            for param_index, param in enumerate(model.parameters()):
                delta_l_k_model_inner_product += torch.sum(self.delta_l_k[param_index] * \
                    local_weight_collector[param_index])

                prox_square += torch.sum(feddyn_alpha / 2 * torch.norm(local_weight_collector[param_index] - \
                    global_weight_collector[param_index])**2)

            output['loss'] = output['loss'] - delta_l_k_model_inner_product + prox_square
            # optimizer.zero_grad()
            output['loss'].backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['max_clip_norm'])
            optimizer.step()

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
        
        # a = model.parameters()
        # b = model.state_dict()
        local_weight_collector = None
        local_weight_collector = list(model.parameters())
        for param_index, param in enumerate(model.parameters()):
            temp = copy.deepcopy(feddyn_alpha * (local_weight_collector[param_index].detach() - \
                global_weight_collector[param_index].detach()))
            self.delta_l_k[param_index] -= temp

        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        return
