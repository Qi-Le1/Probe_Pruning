from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import math
import torch
import random
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
    ServerType,
)

from typing import Any

from models.api import (
    create_model,
    make_batchnorm
)

from optimizer.api import create_optimizer

from data import (
    make_data_loader,
    separate_dataset
)

from .clientBase import ClientBase


class ClientDynamicSgd(ClientBase):

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
        optimizer = create_optimizer(model, 'client')
        self.optimizer_state_dict = optimizer.state_dict()
        self.active = False
    
    @classmethod
    def create_clients(
        cls,
        model: ModelType, 
        data_split: dict[str, dict[int, list[int]]],
    ) -> dict[int, object]:
        '''
        Create clients which organized in dict type
        For DynamicSgd, we only need to create one model to
        speed up the computation.
        The mathematic form of running sgd on one model is
        the same as the create mutiple clients and average
        the gradients. 
        
        Parameters
        ----------
        model: ModelType
        data_split: dict[str, dict[int, list[int]]]

        Returns
        -------
        dict[int, object]
        '''
        clients = [None]
        clients[0] = ClientDynamicSgd(
            client_id=0, 
            model=model, 
            data_split={
                'train': data_split['train'], 
                'test': data_split['test']
            },
        )


       
        return clients

    def check_batch_distribution(
        self,
        input
    ):

        temp_input = input['target'].tolist()

        number_count = []
        total_count = 0
        for i in range(10):
            number_count.append(temp_input.count(i))
        total_count = len(temp_input)
        return number_count, total_count


    def train(
        self, 
        # dataset: DatasetType, 
        data_loader,
        # client_sampler: Any, #instance. A custom Sampler that 
        # yields a list of batch indices at a time can be passed as the batch_sampler argument
        lr: int, 
        metric: MetricType, 
        logger: LoggerType,
        # grad_updates_num: int,
    ) -> None:

        # if grad_updates_num == 0:
        #     return

        model = create_model()
        # print(f'kaishi self.model_state_dict: {self.model_state_dict}')
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = create_optimizer(model, 'client')
        optimizer.load_state_dict(self.optimizer_state_dict)

        model.train(True)


        grads = []
        num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
        for i, input in enumerate(data_loader):

            temp_input = copy.deepcopy(input)


            input = collate(input)
            

            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            optimizer.zero_grad()
            output = model(input)
            # print(f"output_loss: {output['loss']} \n")
            # print(f'output: {output}')
            loss = output['loss']
            loss = loss / num_active_clients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['max_clip_norm'])
            for param_index, param in enumerate(model.parameters()):
                if param_index < len(grads):
                    grads[param_index] += copy.deepcopy(param.grad)
                else:
                    grads.append(copy.deepcopy(param.grad))
                param.grad = None
            if i > 0 and i % (num_active_clients - 1) == 0:
                for param_index, param in enumerate(model.parameters()):
                    param.grad = grads[param_index]
                
                grads = []
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


        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        
        
        return


    
