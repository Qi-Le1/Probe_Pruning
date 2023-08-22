from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import torch
import random
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from collections import defaultdict

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
    create_model,
    make_batchnorm
)
from optimizer.api import create_optimizer
from .serverBase import ServerBase, ClientSampler

from data import (
    separate_dataset,
    make_data_loader
)

class ServerDynamicSgd(ServerBase):

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
    

    def train(
        self,
        dataset: DatasetType,  
        optimizer: OptimizerType, 
        metric: MetricType, 
        logger: LoggerType, 
        global_epoch: int
    ):
        
        logger.safe(True)
        selected_client_ids, num_active_clients = super().select_clients(clients=self.clients) 
        # print(f'dynamicsgd selected_client_ids: {selected_client_ids}')
        start_time = time.time()
        lr = optimizer.param_groups[0]['lr']

        data_loader = None
        client_sampler = ClientSampler(
            batch_size=cfg['client']['batch_size']['train'], 
            data_split=self.clients[0].data_split['train'],
            max_local_gradient_update=cfg['max_local_gradient_update'],
            selected_client_ids=selected_client_ids
        )

        data_loader = make_data_loader(
            dataset={'train': dataset}, 
            tag='client',
            batch_sampler={'train': client_sampler}
        )['train']

        self.clients[0].active = True
        self.clients[0].train(
            data_loader=data_loader,
            lr=lr, 
            metric=metric, 
            logger=logger,
        )

        super().add_log(
            i=0,
            num_active_clients=num_active_clients,
            start_time=start_time,
            global_epoch=global_epoch,
            lr=lr,
            selected_client_ids=selected_client_ids,
            metric=metric,
            logger=logger,
        )

        logger.safe(False)  
        logger.reset()
        self.update_server_model(clients=self.clients) 
        # print(f'jieshu self.server_model_state_dict: {self.server_model_state_dict}')
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
        

