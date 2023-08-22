from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import math
import torch
import random
import itertools
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

from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)

class ServerCombinationSearch(ServerBase):

    def __init__(
        self, 
        model: ModelType,
        clients: dict[int, ClientType],
        dataset: DatasetType,
        communicationMetaData: dict=None
    ) -> None:

        super().__init__(dataset=dataset)
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
        self.client_prob_distribution = {}

        self.genetic_metric_indicator = None

        self.high_freq_clients = None
        self.server_communication_cost_budget = communicationMetaData['server_communication_cost_budget']
        self.server_high_freq_communication_cost_budget = communicationMetaData['server_high_freq_communication_cost_budget']
        self.local_gradient_update_list_to_server_ratio = communicationMetaData['local_gradient_update_list_to_server_ratio']
        # print(f'communicationMetaData: {communicationMetaData}')
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

        a = 5


    

    # def update_min_criteria(self, new_distance, cur_min_distance, clients_indices_indicator, cur_min_clients_indices_indicator):
    #     return new_distance < self.threshold and new_distance < cur_min_distance \
    #         and sum(clients_indices_indicator) <= sum(cur_min_clients_indices_indicator) \
    #             and sum(clients_indices_indicator) >= 1

    

    def cal_genetic_dist_func(self, *args):
        
        # print(f'args: {args}')
        clients_indices_indicator = args[0]
        # print(f'clients_indices_indicator: {clients_indices_indicator}')
        comb_prob = np.array([0 for _ in range(len(self.dataset.classes_counts))])
        total_size = 0
        for i in range(len(clients_indices_indicator)):
            if int(clients_indices_indicator[i]) == 1:
                client_id = self.selected_client_ids[i]
                total_size += len(self.clients[client_id].data_split['train'])

        for i in range(len(clients_indices_indicator)):
            if int(clients_indices_indicator[i]) == 1:
                client_id = self.selected_client_ids[i]
                sub_prob = super().cal_prob_distribution(self.dataset, self.clients[client_id].data_split['train'], client_id)

                ratio = len(self.clients[client_id].data_split['train'])/total_size
                sub_prob = np.array([prob*ratio for prob in sub_prob])
                comb_prob = comb_prob + sub_prob

        res = None
        if self.genetic_metric_indicator == 'KL':
            res = super().cal_KL_divergence(comb_prob, self.global_labels_distribution)
        elif self.genetic_metric_indicator == 'QL':
            res = super().cal_QL(comb_prob, self.global_labels_distribution)

        # if self.update_min_criteria(KL_divergence, self.min_KL, clients_indices_indicator, self.min_clients_list):
        #     # self.min_clients_num = sum(clients_indices_indicator)
        #     self.min_clients_list = copy.deepcopy(clients_indices_indicator)
        #     self.min_KL = KL_divergence
        return res

    def brute_force(
        self, 
        global_labels_distribution, 
        num_clients, 
        selected_client_ids, 
        logger, 
        dataset,
        metric_indicator
    ):
        res = 100
        result_comb = None
        for comb in itertools.combinations(selected_client_ids, num_clients):
            comb_prob = np.array([0 for _ in range(len(dataset.classes_counts))])
            total_size = 0
            for client_id in comb:
                total_size += len(self.clients[client_id].data_split['train'])
            
            for client_id in comb:
                sub_prob = super().cal_prob_distribution(dataset, self.clients[client_id].data_split['train'], client_id)
                
                ratio = len(self.clients[client_id].data_split['train'])/total_size
                sub_prob = np.array([prob*ratio for prob in sub_prob])
                comb_prob = comb_prob + sub_prob

            cur_dist = None
            if metric_indicator == 'KL':
                cur_dist = super().cal_KL_divergence(comb_prob, global_labels_distribution)
            elif metric_indicator == 'QL':
                cur_dist = super().cal_QL(comb_prob, global_labels_distribution)

            if cur_dist < res:
                result_comb = copy.deepcopy(comb)
                res = cur_dist
        return res, result_comb

    def genetic(self, num_clients, selected_client_ids, distance_type):

        self.selected_client_ids = selected_client_ids
        lb = [0 for _ in range(len(selected_client_ids))]
        ub = [1 for _ in range(len(selected_client_ids))]
        precision = [1 for _ in range(len(selected_client_ids))]
        # constraint_ueq = [
        #     lambda x: 1 - sum(x),
        #     lambda x: sum(x) - 3
        #     # lambda x: self.cal_KL_func(x) - threshold
        # ]
        constraint_eq = [
            lambda x: num_clients - sum(x)
        ]
        self.genetic_metric_indicator = distance_type

        ga = GA(func=self.cal_genetic_dist_func, n_dim=len(selected_client_ids), size_pop=50, max_iter=200, prob_mut=0.001, 
                lb=lb, ub=ub, constraint_eq=constraint_eq, precision=precision)
        best_x, self.min_KL = ga.run()

        return self.min_KL[0], best_x

    
    def find_high_freq_group_clients(self, selected_client_ids, permutation_lists, dataset, logger, num_clients):
        self.global_labels_distribution = super().get_global_labels_distribution(dataset)
        # self.max_global_labels_distribution_KL = super().cal_KL_divergence(global_labels_distribution_copy, global_labels_distribution)
        self.dataset = dataset
        self.selected_client_ids = selected_client_ids

        # self.threshold = 1
        # self.min_clients_list = [1 for _ in range(len(selected_client_ids))]
        # self.min_KL = 100
        temp = copy.deepcopy(selected_client_ids)
        temp_2 = copy.deepcopy(selected_client_ids)
        if cfg['cal_communication_cost'] == False:
            if num_clients <= 10:
                start = time.time()
                best_distance, best_combination = self.brute_force(
                    global_labels_distribution=self.global_labels_distribution, 
                    num_clients=num_clients, 
                    selected_client_ids=selected_client_ids, 
                    logger=logger, 
                    dataset=dataset,
                    metric_indicator='KL'
                )
                end = time.time()
                print(f'brute force KL time: {end-start}', flush=True)
                print(f'brute force KL_{num_clients}: {best_distance}', flush=True)
                print(f'brute force KL_comb_{num_clients}: {best_combination}', flush=True)
                logger.append(
                    {
                        f'brute_force_KL_{num_clients}': best_distance,
                        f'brute_force_KL_{num_clients}_time': end-start
                    }, 
                    'train', 
                )

            start = time.time()
            best_genetic_KL_list = []
            best_genetic_KL_combination_list = []
            for i in range(len(permutation_lists)):
                best_distance, best_combination = self.genetic(
                    num_clients=num_clients, 
                    selected_client_ids=permutation_lists[i],
                    distance_type='KL'
                )
                best_genetic_KL_list.append(best_distance)
                best_genetic_KL_combination_list.append(best_combination)
            end = time.time()
            print(f'genetic KL time: {end-start}', flush=True)
            print(f'genetic KL_{num_clients}: {min(best_genetic_KL_list)}', flush=True)
            print(f'genetic KL_comb_{num_clients}: {best_genetic_KL_combination_list[best_genetic_KL_list.index(min(best_genetic_KL_list))]}', flush=True)
            logger.append(
                {
                    f'genetic_KL_{num_clients}': min(best_genetic_KL_list),
                    f'genetic_KL_{num_clients}_time': end-start
                }, 
                'train', 
            )

            start = time.time()
            best_dp_KL_dist = []
            best_dp_KL_combination_list = []
            for i in range(len(permutation_lists)):
                best_distance, best_combination, _, _ = super().dp_combination_search(
                    dataset=dataset,
                    num_clients=num_clients, 
                    selected_client_ids=permutation_lists[i],
                    distance_type='KL',
                )
                # print(f'\n best_dp_KL_{num_clients}: {best_distance}, best_combination_{num_clients}: {best_combination}', flush=True)
                best_dp_KL_dist.append(best_distance)
                best_dp_KL_combination_list.append(best_combination)
            end = time.time()
            print(f'dp KL time: {end-start}', flush=True)
            print(f'dp KL_{num_clients}: {min(best_dp_KL_dist)}', flush=True)
            print(f'dp KL_comb_{num_clients}: {best_dp_KL_combination_list[best_dp_KL_dist.index(min(best_dp_KL_dist))]}', flush=True)
            logger.append(
                {
                    f'best_dp_KL_{num_clients}': min(best_dp_KL_dist),
                    f'best_dp_KL_{num_clients}_time': end-start
                }, 
                'train', 
            )
        elif cfg['cal_communication_cost'] == True:
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
            
            # if cfg['only_high_freq'] == True:
            #     low_freq_clients = []
            # cur_dynamicfl_cost = super().cal_communication_cost(
            #     model_size=cfg['normalized_model_size'],
            #     high_freq_client_num=len(self.high_freq_clients), 
            #     low_freq_client_num=len(low_freq_clients), 
            #     high_freq_communication_times=self.server_high_freq_communication_times, 
            #     low_freq_communication_times=self.server_low_freq_communication_times,
            # )

            # all fedsgd cost
            maximum_cost = super().cal_communication_cost(
                model_size=cfg['normalized_model_size'],
                high_freq_client_num=len(selected_client_ids), 
                low_freq_client_num=0, 
                high_freq_communication_times=cfg['max_local_gradient_update'], 
                low_freq_communication_times=self.server_low_freq_communication_times,
            )

            fedavg_cost = len(selected_client_ids) * 2 * cfg['normalized_model_size']

            
            fedavg_cost_ratio = fedavg_cost / maximum_cost
            # ratio = all_high_freq_dynamicfl_cost / maximum_cost
            logger.append(
                    {
                        f"fedavg_cost_ratio": fedavg_cost_ratio,
                    
                    }, 
                    'train', 
                )
            num_clients = len(selected_client_ids)

            cfg['upload_freq_level'] = {
                # 6: 1,
                # 5: 4,
                # 4: 16,
                # 3: 64,
                # 2: 128,
                # 1: 256

                6: 1,
                5: 4,
                4: 16,
                3: 32,
                2.5: 64,
                2: 128,
                1: 256
            }

            high_freq = float(cfg['number_of_freq_levels'].split('-')[0])
            low_freq = float(cfg['number_of_freq_levels'].split('-')[1])

            low_freq_traverse_list = []
            if high_freq == 6:
                low_freq_traverse_list = [5 ,4, 3, 2.5, 2, 1]
            elif high_freq == 5:
                low_freq_traverse_list = [4, 3, 2.5, 2, 1]
            elif high_freq == 4:
                low_freq_traverse_list = [3, 2.5, 2, 1]
            elif high_freq == 3:
                low_freq_traverse_list = [2.5, 2, 1]

            for key in low_freq_traverse_list:
                temp_times = self.server_low_freq_communication_times / (cfg['upload_freq_level'][key] / cfg['upload_freq_level'][low_freq])
                cur_dynamicfl_cost = super().cal_communication_cost(
                    model_size=cfg['normalized_model_size'],
                    high_freq_client_num=len(self.high_freq_clients), 
                    low_freq_client_num=len(low_freq_clients), 
                    high_freq_communication_times=self.server_high_freq_communication_times, 
                    low_freq_communication_times=temp_times,
                )

                cur_dynamicfl_cost_ratio = cur_dynamicfl_cost / maximum_cost
                logger.append(
                    {
                        f"special_{cfg['server_ratio']}_{cfg['client_ratio']}_{high_freq}-{key}": cur_dynamicfl_cost_ratio,
                    
                    }, 
                    'train', 
                )

                print(f"{cfg['server_ratio']}_{cfg['client_ratio']}_{high_freq}-{key}: {cur_dynamicfl_cost_ratio}", flush=True)
            
            only_high_freq_dynamicfl_cost = cur_dynamicfl_cost = super().cal_communication_cost(
                    model_size=cfg['normalized_model_size'],
                    high_freq_client_num=len(self.high_freq_clients), 
                    low_freq_client_num=0, 
                    high_freq_communication_times=self.server_high_freq_communication_times, 
                    low_freq_communication_times=temp_times,
                )
            
            ratio = only_high_freq_dynamicfl_cost / maximum_cost
            logger.append(
                    {
                        f"special_{cfg['server_ratio']}_{cfg['client_ratio']}_{high_freq}-{low_freq}_1": ratio,
                    
                    }, 
                    'train', 
                )
            print(f"{cfg['server_ratio']}_{cfg['client_ratio']}_{high_freq}-{key}_1: {ratio}", flush=True)
            all_high_freq_dynamicfl_cost = cur_dynamicfl_cost = super().cal_communication_cost(
                    model_size=cfg['normalized_model_size'],
                    high_freq_client_num=len(selected_client_ids), 
                    low_freq_client_num=0, 
                    high_freq_communication_times=self.server_high_freq_communication_times, 
                    low_freq_communication_times=temp_times,
                )
            ratio = all_high_freq_dynamicfl_cost / maximum_cost
            logger.append(
                    {
                        f"special_1-0_1-0_{high_freq}-{low_freq}": ratio,
                    
                    }, 
                    'train', 
                )

            print(f"1-0_1-0_{high_freq}-{low_freq}: {ratio}", flush=True)
            logger.append(
                {
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_maximum_cost_{num_clients}": maximum_cost,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_cost_{num_clients}": cur_dynamicfl_cost,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_fedavg_cost_{num_clients}": fedavg_cost,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_cost_ratio_{num_clients}": cur_dynamicfl_cost_ratio,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_fedavg_cost_ratio_{num_clients}": fedavg_cost_ratio,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_high_freq_number_{num_clients}": len(min_dist_combination),

                }, 
                'train', 
            )
            print(f'best_dp_KL_{cfg["server_ratio"]}_{cfg["client_ratio"]}_ratio_communication_maximum_cost_{num_clients}: {maximum_cost}', flush=True)
            print(f'best_dp_KL_{cfg["server_ratio"]}_{cfg["client_ratio"]}_ratio_communication_cur_dynamicfl_cost_{num_clients}: {cur_dynamicfl_cost}', flush=True)
            print(f'best_dp_KL_{cfg["server_ratio"]}_{cfg["client_ratio"]}_ratio_communication_cur_dynamicfl_cost_ratio_{num_clients}: {cur_dynamicfl_cost_ratio}', flush=True)
            print(f'best_dp_KL_{cfg["server_ratio"]}_{cfg["client_ratio"]}_ratio_communication_cur_fedavg_cost_ratio_{num_clients}: {fedavg_cost_ratio}', flush=True)
            print(f'best_dp_KL_{cfg["server_ratio"]}_{cfg["client_ratio"]}_ratio_communication_cur_dynamicfl_high_freq_number_{num_clients}: {len(min_dist_combination)}', flush=True)


        # min_dist_pos = best_dp_KL_dist.index(min(best_dp_KL_dist))
        # min_dist_combination = best_dp_KL_combination_list[min_dist_pos]

        # high_freq_local_gradient_update_list = self.local_gradient_update_list_to_server_ratio[0][0]
        # low_freq_local_gradient_update_list = self.local_gradient_update_list_to_server_ratio[1][0]

        # # fedsgd
        # if cfg['server_ratio'] == '1-0' and cfg['client_ratio'] == '1-0' and cfg['number_of_freq_levels'] == '6-1':
        #     min_dist_combination = copy.deepcopy(selected_client_ids)
        #     print(f'fedsgd, {min_dist_combination}')
            
        # self.high_freq_clients = copy.deepcopy(min_dist_combination)
        # for client_id in self.high_freq_clients:
        #     self.clients[client_id].local_gradient_update_list = copy.deepcopy(list(high_freq_local_gradient_update_list))
        
        # low_freq_clients = list(set(temp_2) - set(min_dist_combination))
        # for client_id in low_freq_clients:
        #     if cfg['only_high_freq'] == True:
        #         self.clients[client_id].local_gradient_update_list = []
        #     else:
        #         self.clients[client_id].local_gradient_update_list = copy.deepcopy(list(low_freq_local_gradient_update_list))

        # 

            # start = time.time()
            # best_distance, best_combination = self.brute_force(
            #     global_labels_distribution=global_labels_distribution, 
            #     num_clients=num_clients, 
            #     selected_client_ids=selected_client_ids, 
            #     logger=logger, 
            #     dataset=dataset,
            #     metric_indicator='QL'
            # )
            # end = time.time()
            # print(f'brute force QL time: {end-start}', flush=True)
            # print(f'brute force QL_{num_clients}: {best_distance}', flush=True)
            # print(f'brute force QL_comb_{num_clients}: {best_combination}', flush=True)
            # logger.append(
            #     {
            #         f'brute_force_QL_{num_clients}': best_distance,
            #         f'brute_force_QL_{num_clients}_time': end-start
            #     }, 
            #     'train', 
            # )

        # start = time.time()
        # best_genetic_QL_list = []
        # best_genetic_QL_combination_list = []
        # for i in range(len(permutation_lists)):
        #     best_distance, best_combination = self.genetic(
        #         num_clients=num_clients, 
        #         selected_client_ids=permutation_lists[i],
        #         distance_type='QL'
        #     )
        #     best_genetic_QL_list.append(best_distance)
        #     best_genetic_QL_combination_list.append(best_combination)
        # end = time.time()
        # print(f'genetic QL time: {end-start}', flush=True)
        # print(f'genetic QL_{num_clients}: {min(best_genetic_QL_list)}', flush=True)
        # print(f'genetic QL_comb_{num_clients}: {best_genetic_QL_combination_list[best_genetic_QL_list.index(min(best_genetic_QL_list))]}', flush=True)
        # logger.append(
        #     {
        #         f'genetic_QL_{num_clients}': min(best_genetic_QL_list),
        #         f'genetic_QL_{num_clients}_time': end-start
        #     }, 
        #     'train', 
        # )

        # start = time.time()
        # best_dp_KL_dist = []
        # best_dp_KL_combination_list = []
        # for i in range(len(permutation_lists)):
        #     best_distance, best_combination = super().dp(
        #         num_clients=num_clients, 
        #         selected_client_ids=permutation_lists[i],
        #         distance_type='KL',
        #         high_freq_ratio=1
        #     )
        #     # print(f'\n best_dp_KL_{num_clients}: {best_distance}, best_combination_{num_clients}: {best_combination}', flush=True)
        #     best_dp_KL_dist.append(best_distance)
        #     best_dp_KL_combination_list.append(best_combination)
        # end = time.time()
        # print(f'dp KL time: {end-start}', flush=True)
        # print(f'dp KL_{num_clients}: {min(best_dp_KL_dist)}', flush=True)
        # print(f'dp KL_comb_{num_clients}: {best_dp_KL_combination_list[best_dp_KL_dist.index(min(best_dp_KL_dist))]}', flush=True)
        # logger.append(
        #     {
        #         f'best_dp_KL_{num_clients}': min(best_dp_KL_dist),
        #         f'best_dp_KL_{num_clients}_time': end-start
        #     }, 
        #     'train', 
        # )
        return

    def distribute_local_gradient_update_list(self, selected_client_ids: list[int], dataset, logger):
        '''
        distribute local gradient update list to certain selected clients 
        according to the client ratio
        '''
        temp = copy.deepcopy(selected_client_ids)
        # a = self.communicationMetaData['local_gradient_update_list_to_client_ratio']
        
        permutation_lists = super().get_selected_client_ids_permutation_lists(selected_client_ids)

        if cfg['cal_communication_cost'] == True:
            # for size in range(1, len(selected_client_ids)+1):
                # print(size)
            self.find_high_freq_group_clients(
                temp, 
                permutation_lists,
                dataset,
                logger,
                num_clients=len(selected_client_ids)
                # num_clients=min(math.ceil(ratio * len(selected_client_ids)), len(temp))
            )

        
        elif cfg['cal_communication_cost'] == False:
            for size in range(1, len(selected_client_ids)+1):
            # for size in range(1, 2):
                # print(size)
                self.find_high_freq_group_clients(
                    temp, 
                    permutation_lists,
                    dataset,
                    logger,
                    num_clients=size
                    # num_clients=min(math.ceil(ratio * len(selected_client_ids)), len(temp))
                )
        # else:
        #     cur_min_best_dp_KL_dist = super().dp_find_high_freq_group_clients(
        #         temp, 
        #         permutation_lists,
        #         dataset,
        #         logger,
        #         num_clients=len(selected_client_ids)
        #     )
            # if size == 1:
            #     min_best_dp_KL_dist_list.append(cur_min_best_dp_KL_dist)
        # min_best_dp_KL_dist_list.sort()
        # min_best_dp_KL_dist = None
        # for val in min_best_dp_KL_dist_list:
        #     if val != 0:
        #         min_best_dp_KL_dist = val
        #         break
        # print(f'min_best_dp_KL_dist: {min_best_dp_KL_dist}', flush=True)
        # for size in range(1, len(selected_client_ids)+1):
        #     self.find_group_clients_with_smallest_divergence_and_communication_cost(
        #         temp, 
        #         permutation_lists,
        #         dataset,
        #         logger,
        #         num_clients=size,
        #         # min_best_dp_KL_dist=min_best_dp_KL_dist,
        #         local_gradient_update_list_to_client_ratio=local_gradient_update_list_to_client_ratio
        #     )
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
        # super().distribute_server_model_to_clients(
        #     server_model_state_dict=self.server_model_state_dict,
        #     clients=self.clients
        # )

        # overwrite the local_gradient_update_list in selected clients
        # if cfg['select_client_mode'] == 'nonpre':
        self.distribute_local_gradient_update_list(
            selected_client_ids=selected_client_ids,
            dataset=dataset,
            logger=logger,
            # method_comparison=True
        )
        logger.safe(False)
        logger.reset()
        return




# start = time.time()
        # best_dp_QL_dist = []
        # best_dp_QL_combination_list = []
        # for i in range(len(permutation_lists)):
        #     best_distance, best_combination = super().dp(
        #         num_clients=num_clients, 
        #         selected_client_ids=permutation_lists[i],
        #         distance_type='QL'
        #     )
        #     # print(f'\n best_dp_KL_{num_clients}: {best_distance}, best_combination_{num_clients}: {best_combination}', flush=True)
        #     best_dp_QL_dist.append(best_distance)
        #     best_dp_QL_combination_list.append(best_combination)
        # end = time.time()
        # print(f'dp QL time: {end-start}', flush=True)
        # print(f'dp QL_{num_clients}: {min(best_dp_QL_dist)}', flush=True)
        # print(f'dp QL_comb_{num_clients}: {best_dp_QL_combination_list[best_dp_QL_dist.index(min(best_dp_QL_dist))]}', flush=True)
        # logger.append(
        #     {
        #         f'best_dp_QL_{num_clients}': min(best_dp_QL_dist),
        #         f'best_dp_QL_{num_clients}_time': end-start
        #     }, 
        #     'train', 
        # )


        # 
        # selected_client_ids = [i for i in range(30)]
        # start = time.time()
        # import pulp
        # size = 9
        # # possible_tables = [tuple(c) for c in pulp.combination(selected_client_ids, len(selected_client_ids))]
        
        # # A guest must seated at one and only one table
        # # for guest in guests:
        # #     seating_model += (
        # #         pulp.lpSum([x[table] for table in possible_tables if guest in table]) == 1,
        # #         f"Must_seat_{guest}",
        # #     )

        # # possible_tables = [tuple(c) for c in pulp.combination(selected_client_ids, size)]
        # possible_tables = [tuple(c) for c in pulp.combination(selected_client_ids, size)]
        # # create a binary variable to state that a table setting is used
        # x = pulp.LpVariable.dicts(
        #     "table", possible_tables, lowBound=0, upBound=1, cat=pulp.LpInteger
        # )

        # seating_model = pulp.LpProblem("Wedding Seating Model", pulp.LpMinimize)

        # seating_model += pulp.lpSum([self.cal_KL_func(table) * x[table] for table in possible_tables])
        # seating_model += (
        #     pulp.lpSum([self.cal_KL_func(possible_tables, [x[client_id] for client_id in possible_tables])]) <= 0.1,
        #     'threshold'
        # )

        # specify the maximum number of tables
        # seating_model += (
        #     pulp.lpSum([x[table] for table in possible_tables]) == 1,
        #     "Maximum_number_of_tables",
        # )

        # A guest must seated at one and only one table
        # for guest in guests:
        #     seating_model += (
        #         pulp.lpSum([x[table] for table in possible_tables if guest in table]) == 1,
        #         f"Must_seat_{guest}",
        #     )

        # seating_model.solve()
        # end = time.time()
        # print(start, end)
        # print(f"The choosen tables are out of a total of {len(possible_tables)}:")
        # for table in possible_tables:
        #     if x[table].value() == 1.0:
        #         print(table)



        # possible_tables = [client_id for client_id in selected_client_ids]
        # x = pulp.LpVariable.dicts(
        #     "table", possible_tables, lowBound=0, upBound=1, cat=pulp.LpInteger
        # )
        # seating_model = pulp.LpProblem("Wedding Seating Model", pulp.LpMinimize)
        # seating_model += (
        #     pulp.lpSum([self.cal_KL_func(possible_tables, [x[client_id] for client_id in possible_tables])]) <= 0.1,
        #     'threshold'
        # )
        # seating_model += (
        #     pulp.lpSum([x[table] for table in possible_tables]) >= 1,
        #     "Maximum_number_of_tables",
        # )
        a = 5
        # from scipy.optimize import dual_annealing
        # bounds = [[0, 1] for _ in range(len(selected_client_ids))]
        # args=(1,)
        # res = dual_annealing(self.cal_KL_func, bounds=bounds, args=args)
        # print('dual_annealing_res', res)


        # from scipy.optimize import minimize_scalar
        # bounds = [[0, 1] for _ in range(len(selected_client_ids))]
        # args=(1,)
        # res = minimize_scalar(self.cal_KL_func, bounds=bounds, args=args, method='bounded')
        # print('minimize_scalar', res)
        # selected_client_ids = [i for i in range(25)]
        # start_time = time.time()
        # from scipy.optimize import brute
        # ranges = (slice(0, 2, 1),) * len(selected_client_ids)
        # args=(selected_client_ids, 9)
        # res = brute(self.cal_KL_func, full_output=True, ranges=ranges, args=args)
        # end_time = time.time()
        # print(res)
        # print(start_time, end_time)
        # from scipy.optimize import milp
        


        # from sko.GA import GA
        # ga = GA(func=self.cal_KL_func, n_dim=3, size_pop=100, max_iter=500, prob_mut=0.001,
        # lb=[-1, -10, -5], ub=[2, 10, 2], precision=[1e-7, 1e-7, 1])
        # result_comb = None
        # for comb in itertools.combinations(selected_client_ids, num_clients):
        #     comb_prob = np.array([0 for _ in range(len(dataset.classes_counts))])
        #     total_size = 0
        #     for client_id in comb:
        #         total_size += len(self.clients[client_id].data_split['train'])
            
        #     for client_id in comb:
        #         sub_prob = super().cal_prob_distribution(self.clients[client_id].data_split['train'], dataset, client_id)
                
        #         ratio = len(self.clients[client_id].data_split['train'])/total_size
        #         sub_prob = np.array([prob*ratio for prob in sub_prob])
        #         comb_prob = comb_prob + sub_prob
        #         # data_split[client_id]
        #     # print('comb_prob', comb_prob, sum(comb_prob), sum(rel_entr(comb_prob, iid)))
        #     # KL_divergence
        #     if sum(rel_entr(comb_prob, global_labels_distribution)) < KL_divergence:
        #         result_comb = copy.deepcopy(comb)
        #         KL_divergence = sum(rel_entr(comb_prob, global_labels_distribution))
        
        # logger.append(
        #     {'KL_divergence': KL_divergence}, 
        #     'train', 
        # )
