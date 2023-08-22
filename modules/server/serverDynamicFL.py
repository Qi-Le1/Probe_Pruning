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
from sko.GA import GA
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

from utils.api import (
    to_device,  
    collate
)
from optimizer.api import create_optimizer
from .serverBase import ServerBase, ClientSampler
from .serverCombinationSearch import ServerCombinationSearch

from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)

class ServerDynamicFL(ServerBase):

    def __init__(
        self, 
        model: ModelType,
        clients: dict[int, ClientType],
        dataset: DatasetType,
        communicationMetaData: dict=None
    ) -> None:
        ServerBase.__init__(self, dataset=dataset)
        # ServerCombinationSearch.__init__(self, model=model, dataset=dataset, clients=clients, communicationMetaData=communicationMetaData)
        # train dataset
        # self.dataset = dataset
        self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        server_optimizer = create_optimizer(model, 'server')
        self.server_optimizer_state_dict = server_optimizer.state_dict()

        # dict[int, list], record the participated clients
        # at certain local gradient update
        self.dynamic_uploaded_clients = defaultdict(list)
        # dict[int, list], record the model state dict
        # of the participated clients at certain
        # local gradient update
        self.dynamic_iterates = defaultdict(list)
        self.clients = clients
        self.communicationMetaData = communicationMetaData
        self.high_freq_clients = None
        self.server_communication_cost_budget = communicationMetaData['server_communication_cost_budget']
        self.server_high_freq_communication_cost_budget = communicationMetaData['server_high_freq_communication_cost_budget']
        self.local_gradient_update_list_to_server_ratio = communicationMetaData['local_gradient_update_list_to_server_ratio']

        self.server_high_freq_communication_times, self.server_low_freq_communication_times = super().get_high_and_low_freq_communication_time(self.local_gradient_update_list_to_server_ratio)
        self.server_high_freq_communication_cost = super().cal_communication_cost(
            model_size=cfg['normalized_model_size'],
            high_freq_client_num=1, 
            low_freq_client_num=0, 
            high_freq_communication_times=self.server_high_freq_communication_times, 
            low_freq_communication_times=0,
        )

        self.server_low_freq_communication_cost = super().cal_communication_cost(
            model_size=cfg['normalized_model_size'],
            high_freq_client_num=0, 
            low_freq_client_num=1, 
            high_freq_communication_times=0, 
            low_freq_communication_times=self.server_low_freq_communication_times,
        )

    def distribute_dynamic_part(
        self,
        local_gradient_update: int,
    ):
        # return if there is no new server model
        if local_gradient_update not in self.dynamic_uploaded_clients:
            return
        # print(f'distribute_dynamic_part: {self.dynamic_uploaded_clients}')
        model = self.create_model(track_running_stats=False, on_cpu=True)
        model.load_state_dict(self.server_model_state_dict)
        server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        for cur_client_id in self.dynamic_uploaded_clients[local_gradient_update]:
            # print(f'cur_client_id: {cur_client_id}')
            self.clients[cur_client_id].model_state_dict = copy.deepcopy(server_model_state_dict)

        del self.dynamic_uploaded_clients[local_gradient_update]
        return

    def upload_dynamic_part(
        self,
        target_gradent_update: int,
        cur_client_id: int,
    ) -> None:
        '''
        handle dynamic logic, do union operation 
        '''
        self.dynamic_uploaded_clients[target_gradent_update].append(cur_client_id)
        self.dynamic_iterates[target_gradent_update].append(copy.deepcopy(self.clients[cur_client_id].model_state_dict))
        return

    def update_dynamic_part(
        self,
        local_gradent_update: int,
    ):
        with torch.no_grad():
            # return if there is nothing to update
            if local_gradent_update not in self.dynamic_uploaded_clients:
                return
            # print(f'update_dynamic_part: {self.dynamic_uploaded_clients}')
            new_model_parameters_list = self.dynamic_iterates[local_gradent_update]
            # for i in range(len(new_model_parameters_list)):
            #     print(i, new_model_parameters_list[i]['conv1.weight'][0][0])
            if len(new_model_parameters_list) > 0:
                model = super().create_model(track_running_stats=False, on_cpu=True)
                model.load_state_dict(self.server_model_state_dict)
                server_optimizer = create_optimizer(model, 'server')
                server_optimizer.load_state_dict(self.server_optimizer_state_dict)
                server_optimizer.zero_grad()
                # weight = torch.ones(len(new_model_parameters_list))
                # weight = weight / weight.sum()
                weight = []
                for client_id in self.dynamic_uploaded_clients[local_gradent_update]:
                    valid_client = self.clients[client_id]
                    cur_data_size = len(valid_client.data_split['train'])

                    if cfg['server_aggregation'] == 'WA':
                        weight.append(cur_data_size)
                    elif cfg['server_aggregation'] == 'MA':
                        weight.append(1)
                new_weight = [i / sum(weight) for i in weight]
                weight = torch.tensor(new_weight)
                # print(f'weight: {weight}')
                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(new_model_parameters_list)):
                            tmp_v += weight[m] * new_model_parameters_list[m][k]
                        # if k == 'conv1.weight':
                        #     print(f'tmp_v: {tmp_v[0][0]}')
                        v.grad = (v.data - tmp_v).detach()
                server_optimizer.step()
                self.server_optimizer_state_dict = server_optimizer.state_dict()
                self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            
            # delete the data store in 2 dicts when the updating is done
            # del self.dynamic_uploaded_clients[target_gradent_update]
            del self.dynamic_iterates[local_gradent_update]
        return

    def update_server_model(self, clients: dict[int, ClientType]) -> None:
        with torch.no_grad():
            valid_clients = [clients[i] for i in range(len(clients)) if clients[i].active]
            temp = [i for i in range(len(clients)) if clients[i].active ]
            # print(f'valid_clients: {valid_clients}, {temp}')
            if valid_clients:
                model = super().create_model(track_running_stats=False, on_cpu=True)
                model.load_state_dict(self.server_model_state_dict)
                # print(f'update_server_model_state_dict: {self.server_model_state_dict}')
                server_optimizer = create_optimizer(model, 'server')
                server_optimizer.load_state_dict(self.server_optimizer_state_dict)
                server_optimizer.zero_grad()
                # weight = torch.ones(len(valid_clients))
                # weight = weight / weight.sum()

                weight = []
                if cfg['server_aggregation'] == 'WA':
                    for valid_client in valid_clients:
                        cur_data_size = len(valid_client.data_split['train'])

                        weight.append(cur_data_size)
                    new_weight = [i / sum(weight) for i in weight]
                    weight = torch.tensor(new_weight)
                elif cfg['server_aggregation'] == 'MA':
                    weight = torch.ones(len(valid_clients))
                    weight = weight / weight.sum()

                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        # self.server_model_state_dict[k] = v.data.new_zeros(v.size())
                        for m in range(len(valid_clients)):
                            tmp_v += weight[m] * valid_clients[m].model_state_dict[k]
                            # self.server_model_state_dict[k] += weight[m] * valid_clients[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                
                server_optimizer.step()
                self.server_optimizer_state_dict = server_optimizer.state_dict()
                self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                # print('((((')
                # print(f'2_update_server_model_state_dict: {self.server_model_state_dict}')
            for i in range(len(clients)):
                clients[i].active = False
        
        # clean dynamic_uploaded_clients and dynamic_iterates for next round
        self.dynamic_uploaded_clients = defaultdict(list)
        self.dynamic_iterates = defaultdict(list)
        return

    def distribute_local_gradient_update_list(self, selected_client_ids: list[int], dataset, logger):
        '''
        distribute local gradient update list to certain selected clients 
        according to the client ratio
        '''
        temp = copy.deepcopy(selected_client_ids)
        temp_2 = copy.deepcopy(selected_client_ids)

        permutation_lists = super().get_selected_client_ids_permutation_lists(selected_client_ids)

        min_dist, min_dist_combination = super().dp_find_high_freq_group_clients(
            temp, 
            permutation_lists,
            dataset,
            logger,
            num_clients=len(selected_client_ids)
        )

        high_freq_local_gradient_update_list = self.local_gradient_update_list_to_server_ratio[0][0]
        low_freq_local_gradient_update_list = self.local_gradient_update_list_to_server_ratio[1][0]

        # fedsgd
        if cfg['server_ratio'] == '1-0' and cfg['client_ratio'] == '1-0':
            min_dist_combination = copy.deepcopy(selected_client_ids)
            print(f'fedsgd, {min_dist_combination}')
            
        self.high_freq_clients = copy.deepcopy(min_dist_combination)
        for client_id in self.high_freq_clients:
            self.clients[client_id].local_gradient_update_list = copy.deepcopy(list(high_freq_local_gradient_update_list))
        
        low_freq_clients = list(set(temp_2) - set(min_dist_combination))
        for client_id in low_freq_clients:
            if cfg['only_high_freq'] == True:
                self.clients[client_id].local_gradient_update_list = []
            else:
                self.clients[client_id].local_gradient_update_list = copy.deepcopy(list(low_freq_local_gradient_update_list))
        
        cur_dynamicfl_cost = super().cal_communication_cost(
            model_size=cfg['normalized_model_size'],
            high_freq_client_num=len(self.high_freq_clients), 
            low_freq_client_num=len(low_freq_clients), 
            high_freq_communication_times=self.server_high_freq_communication_times, 
            low_freq_communication_times=self.server_low_freq_communication_times,
        )

        # all fedsgd cost
        maximum_cost = super().cal_communication_cost(
            model_size=cfg['normalized_model_size'],
            high_freq_client_num=len(selected_client_ids), 
            low_freq_client_num=0, 
            high_freq_communication_times=cfg['max_local_gradient_update'], 
            low_freq_communication_times=self.server_low_freq_communication_times,
        )

        fedavg_cost = len(selected_client_ids) * 2 * cfg['normalized_model_size']

        cur_dynamicfl_cost_ratio = cur_dynamicfl_cost / maximum_cost
        fedavg_cost_ratio = fedavg_cost / maximum_cost

        num_clients = len(selected_client_ids)
        logger.append(
            {
                f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_maximum_cost_{num_clients}": maximum_cost,
                f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_cost_{num_clients}": cur_dynamicfl_cost,
                f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_fedavg_cost_{num_clients}": fedavg_cost,
                f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_cost_ratio_{num_clients}": cur_dynamicfl_cost_ratio,
                f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_fedavg_cost_ratio_{num_clients}": fedavg_cost_ratio,
            }, 
            'train', 
        )

        return

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
        super().distribute_server_model_to_clients(
            server_model_state_dict=self.server_model_state_dict,
            clients=self.clients
        )

        # overwrite the local_gradient_update_list in selected clients
        # if cfg['select_client_mode'] == 'nonpre':
        self.distribute_local_gradient_update_list(
            selected_client_ids=selected_client_ids,
            dataset=dataset,
            logger=logger
        )
        # for i in range(num_active_clients):
        #     self.clients[selected_client_ids[i]].active = True
        data_loader_list = []
        client_sampler_list = []
        for client_id in selected_client_ids:

            client_sampler = ClientSampler(
                batch_size=cfg['client']['batch_size']['train'], 
                data_split=copy.deepcopy(self.clients[client_id].data_split['train']),
                client_id=client_id,
                max_local_gradient_update=cfg['max_local_gradient_update'],
                high_freq_clients=self.high_freq_clients,
            )
            # dataset_m = separate_dataset(dataset, self.clients[client_id].data_split['train'])
            client_sampler_list.append(client_sampler)
            self.clients[client_id].batch_size = client_sampler.batch_size
            data_loader_list.append(make_data_loader(
                dataset={'train': dataset}, 
                tag='client',
                batch_sampler={'train': client_sampler}
            )['train']) 

        start_time = time.time()
        lr = optimizer.param_groups[0]['lr']

        for local_gradient_update in range(1, cfg['max_local_gradient_update'] + 1):
            # print(f'local_gradient_update: {local_gradient_update}')
            # update the server model parameter using self.dynamic_iterates[target_gradent_update]
            self.update_dynamic_part(local_gradent_update=local_gradient_update)

            self.distribute_dynamic_part(local_gradient_update=local_gradient_update)
                
            for i in range(num_active_clients):
                m = selected_client_ids[i]
                if not self.is_local_gradient_update_valid(
                    local_gradient_update=local_gradient_update,
                    local_gradient_update_list=self.clients[m].local_gradient_update_list
                ):
                    continue
                # print(f'local_gradient_update: {local_gradient_update}')
                # dataset_m = separate_dataset(dataset, self.clients[m].data_split['train'])

                grad_updates_num = self.cal_gradient_updates_num(
                    local_gradient_update=local_gradient_update,
                    local_gradient_update_list=self.clients[m].local_gradient_update_list
                )
                
                self.clients[m].train(
                    # dataset=dataset_list[i],
                    client_sampler=client_sampler_list[i],
                    data_loader=data_loader_list[i], 
                    lr=lr, 
                    metric=metric, 
                    logger=logger,
                    grad_updates_num=grad_updates_num
                )

                # upload the new local model parameter to the self.dynamic_iterates
                self.upload_dynamic_part(
                    target_gradent_update=local_gradient_update+grad_updates_num,
                    cur_client_id=m
                )
                
            super().add_dynamicFL_log(
                local_gradient_update=local_gradient_update,
                start_time=start_time,
                global_epoch=global_epoch,
                lr=lr,
                metric=metric,
                logger=logger,
            )
        
        logger.safe(False)
        logger.reset()
        self.update_server_model(clients=self.clients)
        return

    def cal_gradient_updates_num(
        self,
        local_gradient_update: int,
        local_gradient_update_list: list[int]
    ) -> int:
        '''
        Calculate the local gradient update interval until next update, which
        means current client will update gradient for gradient update num times.

        If local_gradient_update is the last ele in the local_gradient_update_list,
        then we update its gradient to max_local_gradient_update
        '''
        index = local_gradient_update_list.index(local_gradient_update)
        if index == len(local_gradient_update_list) - 1:
            return 0

        return local_gradient_update_list[index+1] - local_gradient_update

    def is_local_gradient_update_valid(
        self,
        local_gradient_update: int,
        local_gradient_update_list: list[int]
    ) -> bool:
        '''
        Check if local gradient update is in local gradient update list
        '''
        return local_gradient_update in local_gradient_update_list

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

       