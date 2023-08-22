from __future__ import annotations


import copy
import datetime
from turtle import update
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
import collections
from torch import autograd
# from torchstat import stat

from utils.api import (
    to_device,  
    collate
)

from typing import Any

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

from data import make_data_loader

from .clientBase import ClientBase


class ClientDynamicFL(ClientBase):

    def __init__(
        self, 
        client_id: int, 
        model: ModelType, 
        data_split: list[int],
        client_local_gradient_update_list: list[int],
        client_communication_cost_budget
    ) -> None:

        super().__init__()
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = create_optimizer(model, 'client')
        self.optimizer_state_dict = optimizer.state_dict()
        # self.local_gradient_update_list=local_gradient_update_list
        self.active = False
        self.batch_size = None

        self.client_local_gradient_update_list = client_local_gradient_update_list
        self.client_communication_cost_budget = client_communication_cost_budget

    @classmethod
    def get_high_and_low_freq_communication_time(cls, local_gradient_update_list_to_client_ratio):
        high_freq_communication_times = 0
        high_freq_ratio = 0
        low_freq_communication_times = float('inf')

        for item in local_gradient_update_list_to_client_ratio:
            if len(item[0]) > high_freq_communication_times:
                high_freq_ratio = item[1][0]
                high_freq_communication_times = len(item[0])
            low_freq_communication_times = min(low_freq_communication_times, len(item[0]))
        return high_freq_communication_times-1, low_freq_communication_times-1, high_freq_ratio

    @classmethod
    def create_communication_meta_data(cls, client_ids: list[int]=None) -> list[int]:
        communicationMetaData = {}
        # print('1_client_ids', client_ids)
        client_ratio_to_update_thresholds = Communication.cal_fix_update_thresholds(
            max_local_gradient_update=cfg['max_local_gradient_update'], 
            client_ratio_to_number_of_freq_levels=cfg['client_ratio_to_number_of_freq_levels']
        )
        local_gradient_update_list_to_client_ratio = Communication.calculate_local_gradient_update_list(
            max_local_gradient_update=cfg['max_local_gradient_update'], 
            client_ratio_to_update_thresholds=client_ratio_to_update_thresholds
        )
        # communicationMetaData['local_gradient_update_dict'] = local_gradient_update_dict
        communicationMetaData['local_gradient_update_list_to_client_ratio'] = local_gradient_update_list_to_client_ratio
        
        server_ratio_to_update_thresholds = Communication.cal_fix_update_thresholds(
            max_local_gradient_update=cfg['max_local_gradient_update'], 
            client_ratio_to_number_of_freq_levels=cfg['server_ratio_to_number_of_freq_levels']
        )
        local_gradient_update_list_to_server_ratio = Communication.calculate_local_gradient_update_list(
            max_local_gradient_update=cfg['max_local_gradient_update'], 
            client_ratio_to_update_thresholds=server_ratio_to_update_thresholds
        )
        communicationMetaData['local_gradient_update_list_to_server_ratio'] = local_gradient_update_list_to_server_ratio

        server_communication_cost_budget = 0
        num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
        for local_gradient_update_list, server_ratio_list in local_gradient_update_list_to_server_ratio:
            for server_ratio in server_ratio_list:
                server_communication_cost_budget += Communication.cal_communication_cost(
                    model_size=cfg['normalized_model_size'],
                    high_freq_client_num=int(num_active_clients*server_ratio),
                    low_freq_client_num=0,
                    high_freq_communication_times=len(local_gradient_update_list)-1,
                    low_freq_communication_times=0
                )
        communicationMetaData['server_communication_cost_budget'] = server_communication_cost_budget

        high_freq, low_freq, high_freq_ratio = cls.get_high_and_low_freq_communication_time(local_gradient_update_list_to_server_ratio)
        server_high_freq_communication_cost_budget = Communication.cal_communication_cost(
            model_size=cfg['normalized_model_size'],
            high_freq_client_num=int(num_active_clients*high_freq_ratio),
            low_freq_client_num=0,
            high_freq_communication_times=high_freq,
            low_freq_communication_times=0
        )
        
        communicationMetaData['server_high_freq_communication_cost_budget'] = server_high_freq_communication_cost_budget
        # print('2_client_ids', client_ids)
        client_index_to_local_gradient_update_list = Communication.distribute_local_gradient_update_list_to_client_ratio(
            client_ids=client_ids, 
            local_gradient_update_list_to_client_ratio=local_gradient_update_list_to_client_ratio, 
        )

        communicationMetaData['client_index_to_local_gradient_update_list'] = client_index_to_local_gradient_update_list

        client_index_to_communication_cost_budget = collections.defaultdict(int)
        for client_index, local_gradient_update_list in client_index_to_local_gradient_update_list.items():
            client_index_to_communication_cost_budget[client_index] = Communication.cal_communication_cost(
                model_size=cfg['normalized_model_size'],
                high_freq_client_num=1,
                low_freq_client_num=0,
                high_freq_communication_times=len(local_gradient_update_list)-1,
                low_freq_communication_times=0
            )
        communicationMetaData['client_index_to_communication_cost_budget'] = client_index_to_communication_cost_budget
       
        return communicationMetaData

    @classmethod
    def create_clients(
        cls,
        model: ModelType, 
        data_split: dict[str, dict[int, list[int]]],
        communicationMetaData
    ) -> dict[int, ClientType]:
        client_ids = torch.arange(cfg['num_clients'])
        clients = [None for _ in range(cfg['num_clients'])]
        # communicationMetaData = cls.create_communication_meta_data(client_ids=client_ids)
        for m in range(len(clients)):
            clients[m] = ClientDynamicFL(
                client_id=client_ids[m], 
                model=model, 
                data_split={
                    'train': data_split['train'][m], 
                    'test': data_split['test'][m]
                },
                client_local_gradient_update_list=communicationMetaData['client_index_to_local_gradient_update_list'][m],
                client_communication_cost_budget=communicationMetaData['client_index_to_communication_cost_budget'][m]
            )
        
        return clients

    def total_params_num(self, model: ModelType):
        total_num = 0
        for k, v in model.named_parameters():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                total_num += v.numel()
        return total_num

    def train(
        self, 
        # dataset: DatasetType, 
        # client_sampler,
        client_sampler,
        data_loader,
        # client_id,
        lr: int, 
        metric: MetricType, 
        logger: LoggerType,
        grad_updates_num: int
    ) -> None:

        if grad_updates_num == 0:
            raise ValueError('grad_updates_num must > 0')
            return

        # print(f"data_split_m: {self.data_split['train']}")
        model = create_model()
        # print(f'kaishi self.model_state_dict: {self.model_state_dict}')
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        # print(f'optimizer_state_dict: {self.optimizer_state_dict}')
        optimizer = create_optimizer(model, 'client')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)


        cur_grad_updates_num = 1
        while cur_grad_updates_num <= grad_updates_num:

            for i, input in enumerate(data_loader): 
                input = collate(input)
                
                # if cur_grad_updates_num == 1:
                #     print(f'{input}')
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                # with autograd.detect_anomaly():
                output = model(input)
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
                cur_grad_updates_num += 1                      
                if cur_grad_updates_num == grad_updates_num + 1:
                    break

        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return


class Communication:
    '''
    Class to handle communication issue in DynamicFL
    '''
    @classmethod
    def cal_communication_cost(
        cls, 
        model_size, 
        high_freq_client_num, 
        low_freq_client_num, 
        high_freq_communication_times, 
        low_freq_communication_times
    ):
        return model_size * (high_freq_client_num * high_freq_communication_times * 2 + low_freq_client_num * low_freq_communication_times * 2)
    
    # TODO: dynamic/fix client raio
    @classmethod
    def cal_fix_update_thresholds(
        cls,
        max_local_gradient_update: int,
        client_ratio_to_number_of_freq_levels: dict[float, list[int]]
    ) -> dict[float, list[int]]:
        '''
        Calculate update_threshold
        based on max_local_gradient_update and number_of_freq_levels
        
        Parameters
        ----------
        client_id : list[int]
        max_local_gradient_update : int
        ratio_to_number_of_freq_levels : dict[float, list[int]]
            The key is the ratio of distributing client to each number_of_freq_levels
            The value is the number_of_freq_levels

        Returns
        -------
        list[int]

        Notes
        -----
        update_threshold = int(max_local_gradient_update/number_of_freq_levels)
        '''
        client_ratio_to_update_thresholds = collections.defaultdict(list)
        for ratio, number_of_freq_levels in client_ratio_to_number_of_freq_levels.items():
            for number in number_of_freq_levels:
                freq = int(cfg['upload_freq_level'][number])
                client_ratio_to_update_thresholds[ratio].append(
                    min(freq, cfg['max_local_gradient_update'])
                )

        return client_ratio_to_update_thresholds

    @classmethod
    def cal_local_gradient_update_list(
        cls,
        update_threshold: int
    ) -> int:
        '''
        calculate which local_gradient_update we need to go to inner loop
        of dynamicFL.

        Example:
        If the cur_local_gradient_update == 5, then local_gradient_update_list 
        is [1, 6, 11]. 
        '''
        cur_local_gradient_update = 1
        local_gradient_update_list = []
        while cur_local_gradient_update < cfg['max_local_gradient_update']:
            local_gradient_update_list.append(cur_local_gradient_update)
            cur_local_gradient_update += update_threshold
        
        local_gradient_update_list.append(cfg['max_local_gradient_update']+1)
        return local_gradient_update_list

    @classmethod
    def distribute_local_gradient_update_list_to_client_ratio(
        cls,
        client_ids: list[int],
        local_gradient_update_list_to_client_ratio: list[tuple[list, list]],
    ):
        '''
        Distribute list[local_gradient_update] for each client based on
        update_threshold
        local_gradient_update indicates that the client needs to enter
        dynamicFL algo
        
        Parameters
        ----------
        client_id : list[int]
        max_local_gradient_update : int
        ratio_to_update_thresholds : dict[float, list[int]]
            The key is the ratio of distributing client to each update_thresholds
            The value is the update_thresholds

        Returns
        -------
        client_index_to_local_gradient_update_list : list[list[int]]. index are the same as the index of client_ids, elements represents
            the list of local_gradient_update

        Notes
        -----
        Minimum number of uploads is 1(client must upload once in
        each local gradient update cycle)
        Maximum number of uploads is max_local_gradient_update
        '''
        temp_client_ids = copy.deepcopy(client_ids)
        temp_client_ids = temp_client_ids.tolist()

        client_index_to_local_gradient_update_list = collections.defaultdict(list)
        for local_gradient_update_list, client_ratio_list in local_gradient_update_list_to_client_ratio:
            for client_ratio in client_ratio_list:

                selected_client_ids = random.sample(
                    temp_client_ids, 
                    min(math.ceil(client_ratio * len(client_ids)), len(temp_client_ids))
                )
                for index in selected_client_ids:
                    client_index_to_local_gradient_update_list[index] = copy.deepcopy(local_gradient_update_list)

                temp_client_ids = list(set(temp_client_ids) - set(selected_client_ids))

        return client_index_to_local_gradient_update_list
    
    @classmethod
    def calculate_local_gradient_update_list(
        cls,
        max_local_gradient_update: int,
        client_ratio_to_update_thresholds: dict[float, list[int]]
    ) -> list[tuple[list, list]]:
        '''
        calculate local gradient update list based on update_threshold
        local_gradient_update indicates that the client needs to enter
        dynamicFL algo
        
        Parameters
        ----------
        max_local_gradient_update : int
        ratio_to_update_thresholds : dict[float, list[int]]
            The key is the ratio of distributing client to each update_thresholds
            The value is the update_thresholds

        Returns
        -------
        local_gradient_update_list_to_client_ratio : list[tuple[list, list]]

        Notes
        -----
        Minimum number of uploads is 1(client must upload once in
        each local gradient update cycle)
        Maximum number of uploads is max_local_gradient_update
        '''
        # local_gradient_update_dict = collections.defaultdict(list)
        local_gradient_update_list_to_client_ratio = collections.defaultdict(list)
        for ratio, update_threshold in client_ratio_to_update_thresholds.items():
            for threshold in update_threshold:
                threshold = min(
                    max_local_gradient_update, 
                    max(1, threshold)
                )
                
                local_gradient_update_list = cls.cal_local_gradient_update_list(
                    update_threshold=threshold
                )

                # local_gradient_update_dict[ratio].append(local_gradient_update_list)
                local_gradient_update_list_to_client_ratio[tuple(local_gradient_update_list)].append(ratio)
        local_gradient_update_list_to_client_ratio = sorted(
            local_gradient_update_list_to_client_ratio.items(),
            key=lambda x:-len(x[0])
        )


        return local_gradient_update_list_to_client_ratio
        # return local_gradient_update_dict, local_gradient_update_list_to_client_ratio
    
    @classmethod
    def cut_low_frequency(cls, high_freq_update_list, local_gradient_update_list_to_client_ratio, index):
        if index == len(local_gradient_update_list_to_client_ratio):
            return None

        low_freq_update_list = local_gradient_update_list_to_client_ratio[index][0]
        for j in range(len(low_freq_update_list)-2, -1, -1):
            last_entry = low_freq_update_list[j]
            if last_entry in high_freq_update_list:
                local_gradient_update_list_to_client_ratio[index][0] = copy.deepcopy(low_freq_update_list[:j+1])
        
        # can change high_freq_update_list for further cutting
        cls.cut_low_frequency(high_freq_update_list, local_gradient_update_list_to_client_ratio, index+1)
        return
        


    @classmethod
    def cal_communication_budget(
        cls,
        model_size: int,
        number_of_freq_levels: list[int]
    ) -> list[int]:
        '''
        Calculate communication budget
        based on model size and number_of_freq_levels
        
        Parameters
        ----------
        model_size : int,
        number_of_freq_levels : list[int]

        Returns
        -------
        list[int]

        Notes
        -----
        communication budget = number_of_freq_levels * 2 * model_size
        '''
        communication_budget = [i * 2 * model_size for i in number_of_freq_levels]
        return communication_budget