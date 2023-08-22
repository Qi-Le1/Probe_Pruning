from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import math
import time
import torch
import random
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from collections import defaultdict
from scipy.special import rel_entr

from _typing import (
    ModelType,
    ClientType,
    DatasetType
)

from utils.api import (
    to_device,  
    collate
)

from models.api import (
    create_model,
    make_batchnorm
)

from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)

from optimizer.api import create_optimizer

class ServerBase:

    def __init__(
        self,
        dataset
    ) -> None:
        # dataset is train dataset
        self.fix_order_picking = -1
        # self.dataset = dataset
        self.client_prob_distribution = {}
        self.group_clients_prob_distribution = np.array([0 for _ in range(len(dataset.classes_counts))])
        self.high_freq_clients = {}
        return
    
    def create_model(self, track_running_stats=False, on_cpu=False):
        return create_model(track_running_stats=track_running_stats, on_cpu=on_cpu)

    def create_test_model(
        self,
        model_state_dict,
        batchnorm_dataset
    ) -> object:

        model = create_model()
        model.load_state_dict(model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'server')
        # print(f'test_mode;: {test_model.state_dict()}')
        return test_model

    def distribute_server_model_to_clients(
        self,
        server_model_state_dict,
        clients
    ) -> None:

        model = self.create_model(track_running_stats=False)
        model.load_state_dict(server_model_state_dict)
        server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        for m in range(len(clients)):
            if clients[m].active:
                clients[m].model_state_dict = copy.deepcopy(server_model_state_dict)
        return

    # def cal_client_prob_distribution(self, dataset, data_split, client_id):
    #     if client_id in self.client_prob_distribution:
    #         return self.client_prob_distribution[client_id]

    #     target_list = np.array([dataset[index]['target'].item() for index in data_split])
    #     sub_prob = []
    #     for i in range(len(dataset.classes_counts)):
    #         sub_prob.append(sum(target_list == i)/len(target_list))
    #     # sub_prob = [sum(target_list == i)/len(target_list) for i in range(len(dataset.classes_counts))]
    #     # for i in range(len(sub_prob)):
    #     #     if sub_prob[i] == 0:
    #     #         # prob_list[i] = 1e-5
    #     #         sub_prob[i] = 1e-8
    #     self.client_prob_distribution[client_id] = sub_prob
    #     return np.array(sub_prob)

    # def cal_active_clients_prob_distribution(self, dataset, selected_client_ids):
        
    #     total_size = 0
    #     # for client_id in selected_client_ids:
    #     for client_id in selected_client_ids:
    #         total_size += len(self.clients[client_id].data_split['train'])

    #     self.group_clients_prob_distribution = np.array([0 for _ in range(len(dataset.classes_counts))])
    #     for client_id in selected_client_ids:
    #         # calculate client prob distribution
    #         sub_prob = self.cal_client_prob_distribution(dataset, self.clients[client_id].data_split['train'], client_id)

    #         # calculate active clients prob distribution
    #         ratio = len(self.clients[client_id].data_split['train'])/total_size
    #         sub_prob = np.array([prob*ratio for prob in sub_prob])
    #         self.group_clients_prob_distribution = self.group_clients_prob_distribution + sub_prob

    #     return None


    def add_log(
        self,
        i,
        num_active_clients,
        start_time,
        global_epoch,
        lr,
        selected_client_ids,
        metric,
        logger
    ) -> None:
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            global_epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = global_epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['server']['num_epochs'] - global_epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                            'Train Epoch (C): {}({:.0f}%)'.format(global_epoch, exp_progress),
                            'Learning rate: {:.6f}'.format(lr),
                            'ID: {}({}/{})'.format(selected_client_ids[i], i + 1, num_active_clients),
                            'Global Epoch Finished Time: {}'.format(global_epoch_finished_time),
                            'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']), flush=True)

    def add_dynamicFL_log(
        self,
        local_gradient_update,
        start_time,
        global_epoch,
        lr,
        # selected_client_ids,
        metric,
        logger
    ) -> None:
        if local_gradient_update % int((cfg['max_local_gradient_update'] * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (local_gradient_update + 1)
            global_epoch_finished_time = datetime.timedelta(seconds=_time * (cfg['max_local_gradient_update'] - local_gradient_update - 1))
            exp_finished_time = global_epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['server']['num_epochs'] - global_epoch) * _time * cfg['max_local_gradient_update']))
            exp_progress = 100. * local_gradient_update / cfg['max_local_gradient_update']
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                            'Train Epoch (C): {}({:.0f}%)'.format(global_epoch, exp_progress),
                            'Learning rate: {:.6f}'.format(lr),
                            # 'ID: {}({}/{})'.format(selected_client_ids[i], i + 1, cfg['max_local_gradient_update']),
                            'Global Epoch Finished Time: {}'.format(global_epoch_finished_time),
                            'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']), flush=True)

        return

    def select_clients(
        self, clients: dict[int, ClientType]
    ) -> tuple[list[int], int]:

        num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
        selected_client_ids = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist() 

        if cfg['algo_mode'] != 'dynamicsgd':
            for i in range(num_active_clients):
                clients[selected_client_ids[i]].active = True
        
        return selected_client_ids, num_active_clients
    
    def update_server_model(self, clients: dict[int, ClientType]) -> None:
        with torch.no_grad():
            valid_clients = [clients[i] for i in range(len(clients)) if clients[i].active]
            if valid_clients:
                model = self.create_model(track_running_stats=False, on_cpu=True)
                model.load_state_dict(self.server_model_state_dict)
                server_optimizer = create_optimizer(model, 'server')
                server_optimizer.load_state_dict(self.server_optimizer_state_dict)
                server_optimizer.zero_grad()
                # weight = torch.ones(len(valid_clients))
                # weight = weight / weight.sum()

                weight = []

                for valid_client in valid_clients:
                    if cfg['server_aggregation'] == 'WA':
                        weight.append(len(valid_client.data_split['train']))
                    elif cfg['server_aggregation'] == 'MA':
                        weight.append(1)
                new_weight = [i / sum(weight) for i in weight]
                weight = torch.tensor(new_weight)
                
                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_clients)):
                            tmp_v += weight[m] * valid_clients[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                server_optimizer.step()
                self.server_optimizer_state_dict = server_optimizer.state_dict()
                self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

            for i in range(len(clients)):
                clients[i].active = False
        return
    
    def combine_test_dataset(
        self,
        num_active_clients: int,
        clients: dict[int, ClientType],
        selected_client_ids: list[int],
        dataset: DatasetType
    ) -> DatasetType:  
        '''
        combine the datapoint index for selected clients
        and return the dataset
        '''
        combined_datapoint_idx = []
        for i in range(num_active_clients):
            m = selected_client_ids[i]
            combined_datapoint_idx += clients[m].data_split['test']

        # dataset: DatasetType
        dataset = separate_dataset(dataset, combined_datapoint_idx)
        return dataset

    def evaluate_trained_model(
        self,
        dataset,
        batchnorm_dataset,
        logger,
        metric,
        global_epoch,
        server_model_state_dict
    ):  
        data_loader = make_data_loader(
            dataset={'test': dataset}, 
            tag='server'
        )['test']

        model = self.create_test_model(
            model_state_dict=server_model_state_dict,
            batchnorm_dataset=batchnorm_dataset
        )

        # print(f'test_server_model_state_dict: {server_model_state_dict}')
        # print(f'test_model_state_dict: {model.state_dict()}')
        logger.safe(True)
        with torch.no_grad():
            model.train(False)
            for i, input in enumerate(data_loader):

                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])

                output = model(input)
                # print(f'test_output_2: {output}')
                evaluation = metric.evaluate(
                    metric.metric_name['test'], 
                    input, 
                    output
                )
                # print(f'test_evaluation: {evaluation}')
                logger.append(
                    evaluation, 
                    'test', 
                    input_size
                )
                
            info = {
                'info': [
                    'Model: {}'.format(cfg['model_tag']), 
                    'Test Epoch: {}({:.0f}%)'.format(global_epoch, 100.)
                ]
            }
            logger.append(info, 'test', mean=False)
            print(logger.write('test', metric.metric_name['test']), flush=True)
        logger.safe(False)
        return

    def cal_KL_divergence(self, comb_prob, global_labels_distribution):
        return sum(rel_entr(comb_prob, global_labels_distribution))
    
    def cal_QL(self, list_1, list_2):
        return sum([(item[0]-item[1])**2 for item in zip(list_1, list_2)]) 

    def cal_prob_distribution(self, dataset, data_split, client_id):
        if client_id in self.client_prob_distribution:
            return self.client_prob_distribution[client_id]
        target_list = np.array([dataset[index]['target'].item() for index in data_split])
        sub_prob = []
        for i in range(len(dataset.classes_counts)):
            sub_prob.append(sum(target_list == i)/len(target_list))
        self.client_prob_distribution[client_id] = sub_prob
        return np.array(sub_prob)
    
    def cal_clients_communication_cost(
        self, 
        model_size, 
        client_ids
    ):
        return sum([self.clients[client_id].client_communication_cost_budget for client_id in client_ids])
    
    def cal_dp_dist_func(self, dataset, client_ids, metric_indicator):
        comb_prob = np.array([0 for _ in range(len(dataset.classes_counts))])
        total_size = 0

        for client_id in client_ids:  
            total_size += len(self.clients[client_id].data_split['train'])

        for client_id in client_ids:
            sub_prob = self.cal_prob_distribution(dataset, self.clients[client_id].data_split['train'], client_id)
            ratio = len(self.clients[client_id].data_split['train'])/total_size
            sub_prob = np.array([prob*ratio for prob in sub_prob])
            comb_prob = comb_prob + sub_prob

        res = None
        if metric_indicator == 'KL':
            res = self.cal_KL_divergence(comb_prob, self.global_labels_distribution)
        elif metric_indicator == 'QL':
            res = self.cal_QL(comb_prob, self.global_labels_distribution)
        return res
    
    def dp_combination_search(self, dataset, num_clients, selected_client_ids, distance_type):

        each_item = {
            'distance': float('inf'),
            'client_ids': [],
        }
        dp_res = [[copy.deepcopy(each_item) for _ in range(num_clients+1)] for _ in range(len(selected_client_ids)+1)]
        for i in range(1, len(selected_client_ids)+1):
            for j in range(1, num_clients+1):
                if i < j:
                    dp_res[i][j] = copy.deepcopy(dp_res[i-1][j])
                else:
                    dp_res[i][j] = copy.deepcopy(dp_res[i-1][j])
                    temp = copy.deepcopy(dp_res[i-1][j-1])
                    temp['client_ids'] += [selected_client_ids[i-1]]
                    temp['distance'] = self.cal_dp_dist_func(
                        dataset=dataset,
                        client_ids=temp['client_ids'],
                        metric_indicator=distance_type
                    )

                    if temp['distance'] < dp_res[i][j]['distance'] and len(temp['client_ids']) == j:
                        dp_res[i][j] = copy.deepcopy(temp)


                
            # a = 5
        min_distance = dp_res[-1][num_clients]['distance']
        min_client_ids = dp_res[-1][num_clients]['client_ids']
        # print(f'num_clients: {num_clients}, min_distance: {min_distance}, min_client_ids: {min_client_ids}')
        # if count_smaller_than:
        min_distance_for_all = float('inf')
        min_client_ids_for_all = []
        # for i in range(len(dp_res)):
        for j in range(len(dp_res[0])):
            if dp_res[-1][j]['distance'] < min_distance_for_all:
                min_distance_for_all = dp_res[-1][j]['distance']
                min_client_ids_for_all = dp_res[-1][j]['client_ids']
        return min_distance, min_client_ids, min_distance_for_all, min_client_ids_for_all
    
    def dp(self, dataset, num_clients, selected_client_ids, distance_type):

        each_item = {
            'distance': float('inf'),
            'client_ids': [],
        }
        dp_res = [[copy.deepcopy(each_item) for _ in range(num_clients+1)] for _ in range(len(selected_client_ids)+1)]
        for i in range(1, len(selected_client_ids)+1):
            for j in range(1, num_clients+1):
                if i < j:
                    dp_res[i][j] = copy.deepcopy(dp_res[i-1][j])
                else:
                    dp_res[i][j] = copy.deepcopy(dp_res[i-1][j])
                    temp = copy.deepcopy(dp_res[i-1][j-1])
                    temp['client_ids'] += [selected_client_ids[i-1]]
                    temp['distance'] = self.cal_dp_dist_func(
                        dataset=dataset,
                        client_ids=temp['client_ids'],
                        metric_indicator=distance_type
                    )
                    # add communication cost
                    cur_total_communication_cost = self.cal_clients_communication_cost(
                        model_size=cfg['normalized_model_size'], 
                        client_ids=temp['client_ids'],
                    )

                    cur_client_communication_cost_budget = self.clients[selected_client_ids[i-1]].client_communication_cost_budget
                    # print(f'cur_client_communication_cost_budget: {cur_client_communication_cost_budget}')
                    # print(f'cur_total_communication_cost: {cur_total_communication_cost}')

                    # check client's communication cost
                    if cur_client_communication_cost_budget < self.server_high_freq_communication_cost:
                        continue
                    # check server's communication cost
                    if self.server_high_freq_communication_cost_budget < cur_total_communication_cost:
                        continue
                    # print(f'cur_communication_cost: {cur_communication_cost}')
                    # temp['distance'] += cur_communication_cost
                    if temp['distance'] < dp_res[i][j]['distance'] and len(temp['client_ids']) == j:
                        dp_res[i][j] = copy.deepcopy(temp)
 
                
            # a = 5
        min_distance = dp_res[-1][num_clients]['distance']
        min_client_ids = dp_res[-1][num_clients]['client_ids']
        # print(f'num_clients: {num_clients}, min_distance: {min_distance}, min_client_ids: {min_client_ids}')
        # if count_smaller_than:
        min_distance_for_all = float('inf')
        min_client_ids_for_all = []
        # for i in range(len(dp_res)):
        for j in range(len(dp_res[0])):
            if dp_res[-1][j]['distance'] < min_distance_for_all:
                min_distance_for_all = dp_res[-1][j]['distance']
                min_client_ids_for_all = dp_res[-1][j]['client_ids']
        return min_distance, min_client_ids, min_distance_for_all, min_client_ids_for_all

    def dp_find_high_freq_group_clients(
            self,
            temp, 
            permutation_lists,
            dataset,
            logger,
            num_clients,
            # min_best_dp_KL_dist,
            # local_gradient_update_list_to_client_ratio
        ):


            self.global_labels_distribution = self.get_global_labels_distribution(dataset)
            # for i in range(len(high_freq_ratio_list)):
            start = time.time()
            best_dp_KL_dist = []
            best_dp_KL_combination_list = []
            for j in range(len(permutation_lists)):
                _, _, best_distance, best_combination = self.dp(
                    dataset=dataset,
                    num_clients=num_clients, 
                    selected_client_ids=permutation_lists[j],
                    distance_type='KL',
                )
                # print(f'\n best_dp_KL_{num_clients}: {best_distance}, best_combination_{num_clients}: {best_combination}', flush=True)
                best_dp_KL_dist.append(best_distance)
                best_dp_KL_combination_list.append(best_combination)
            end = time.time()
            min_dist_pos = best_dp_KL_dist.index(min(best_dp_KL_dist))
            min_dist_combination = best_dp_KL_combination_list[min_dist_pos]

            logger.append(
                {
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cost_{num_clients}": min(best_dp_KL_dist),
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cost_{num_clients}_time": end-start,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_combination_size_{num_clients}": len(min_dist_combination)
                }, 
                'train', 
            )

            
            return min_dist_pos, min_dist_combination
    
    def get_global_labels_distribution(self, dataset):
        total_dp = 0
        for val in dataset.classes_counts.values():
            total_dp += val
        global_labels_distribution = [0 for _ in range(len(dataset.classes_counts))]
        global_labels_distribution_copy = copy.deepcopy(global_labels_distribution)
        for key, val in dataset.classes_counts.items():
            global_labels_distribution[key] = val/total_dp
        return global_labels_distribution


    def get_selected_client_ids_permutation_lists(self, selected_client_ids):
        permutation_lists = []
        for i in range(cfg['dp_ensemble_times']):
            temp = copy.deepcopy(selected_client_ids)
            random.shuffle(temp)
            permutation_lists.append(copy.deepcopy(temp))
        return permutation_lists
    
    def cal_communication_cost(
        self, 
        model_size, 
        high_freq_client_num, 
        low_freq_client_num, 
        high_freq_communication_times, 
        low_freq_communication_times
    ):
        return model_size * (high_freq_client_num * high_freq_communication_times * 2 + low_freq_client_num * low_freq_communication_times * 2)

    def get_high_and_low_freq_communication_time(self, local_gradient_update_list_to_client_ratio):
        high_freq_communication_times = 0
        low_freq_communication_times = float('inf')

        for item in local_gradient_update_list_to_client_ratio:
            high_freq_communication_times = max(high_freq_communication_times, len(item[0]))
            low_freq_communication_times = min(low_freq_communication_times, len(item[0]))
        return high_freq_communication_times-1, low_freq_communication_times-1
        
class ClientSampler(torch.utils.data.Sampler):
    def __init__(self, 
        batch_size, 
        data_split, 
        client_id=None, 
        max_local_gradient_update=250, 
        selected_client_ids=None,
        high_freq_clients=None,
        group_clients_prob_distribution=None,
        cur_client_prob_distribution=None,
        dataset=None
    ):
        self.batch_size = batch_size
        self.data_split = data_split
        self.max_local_gradient_update = max_local_gradient_update
        self.client_id = client_id
        self.selected_client_ids = selected_client_ids
        self.high_freq_clients = high_freq_clients
        self.group_clients_prob_distribution = group_clients_prob_distribution
        self.cur_client_prob_distribution = cur_client_prob_distribution
        # self.dataset = dataset
        self.reset()
        self.start = 0
        self.end = len(self.idx)


    def extend_data_split(
        self, 
        local_gradient_update, 
        batch_size,
        client_id=None
    ):
        # sample without replacement
        if cfg['algo_mode'] == 'dynamicsgd':
            total_data_size = local_gradient_update * batch_size
            new_data_split = []
            while len(new_data_split) <= total_data_size:
                random.shuffle(self.data_split[client_id])
                new_data_split.extend(copy.deepcopy(self.data_split[client_id]))
        else:
            total_data_size = local_gradient_update * batch_size
            new_data_split = []
            while len(new_data_split) <= total_data_size:
                random.shuffle(self.data_split)
                new_data_split.extend(copy.deepcopy(self.data_split))
        return new_data_split

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start < self.end:
            res = self.start
            self.start += 1
            return self.idx[res]
        else:
            self.start = 0
            raise StopIteration

    def __len__(self):
        return len(self.idx)
    
    def reset(self):
        # self.data_split_ = copy.deepcopy(self.data_split)
        self.idx = []
        if cfg['algo_mode'] == 'dynamicsgd':
            print('zhelijinlaile')
            for client_id in self.selected_client_ids:
                self.data_split[client_id] = self.extend_data_split(
                    local_gradient_update=self.max_local_gradient_update, 
                    batch_size=self.batch_size,
                    client_id=client_id
            ) 
            start = 0
            while self.max_local_gradient_update > 0:
                batch_idx = []
                for client_id in self.selected_client_ids:
 
                    chosen_eles = copy.deepcopy(self.data_split[client_id][start: start+self.batch_size])  
                    # batch_idx.extend(chosen_eles)
                    self.idx.append(chosen_eles)
                # self.idx.append(batch_idx)
                self.max_local_gradient_update -= 1
                start += self.batch_size
            a = 5
        elif cfg['algo_mode'] == 'dynamicfl' or cfg['algo_mode'] == 'dynamicavg':
            start = 0
            self.data_split = self.extend_data_split(
                local_gradient_update=self.max_local_gradient_update, 
                batch_size=self.batch_size
            )
            while self.max_local_gradient_update > 0:
                batch_idx = []
                chosen_eles = copy.deepcopy(self.data_split[start: start+self.batch_size])   
                batch_idx.extend(chosen_eles)
                self.idx.append(batch_idx)
                self.max_local_gradient_update -= 1
                start += self.batch_size
        elif cfg['algo_mode'] == 'fedavg' or cfg['algo_mode'] == 'scaffold' \
            or cfg['algo_mode'] == 'fedprox' or cfg['algo_mode'] == 'feddyn' \
            or cfg['algo_mode'] == 'fedensemble' or cfg['algo_mode'] == 'fedgen' or cfg['algo_mode'] == 'fednova':

            self.batch_size = min(self.batch_size, len(self.data_split))
            cur_local_gradient_update = int(cfg['local_epoch'] * len(self.data_split) / self.batch_size)
            start = 0
            self.data_split = self.extend_data_split(
                local_gradient_update=cur_local_gradient_update, 
                batch_size=self.batch_size
            )
            while cur_local_gradient_update > 0:
                batch_idx = []
                chosen_eles = copy.deepcopy(self.data_split[start: start+self.batch_size])      
                batch_idx.extend(chosen_eles)
                self.idx.append(batch_idx)
                cur_local_gradient_update -= 1
                start += self.batch_size
        else:
            raise ValueError('wrong algo mode')
        return



    def reweight_local_data_prob(self):
        start = time.time()
        client_prob = np.array(self.cur_client_prob_distribution) 
        # a = client_prob != 0
        group_clients_correspoding_prob = np.zeros(len(self.group_clients_prob_distribution))
        # group_clients_correspoding_prob = np.array([0 for _ in range(len(self.group_clients_prob_distribution))])
        # ceshi = self.group_clients_prob_distribution
        for i in range(len(client_prob)):
            # print(client_prob[i])
            if client_prob[i] > 0:
                # print('yes', self.group_clients_prob_distribution[i])
                group_clients_correspoding_prob[i] = copy.deepcopy(self.group_clients_prob_distribution[i])
                # print('zz', group_clients_correspoding_prob[i])
        # print(f'1111: {group_clients_correspoding_prob}')
        group_clients_correspoding_prob = group_clients_correspoding_prob / sum(group_clients_correspoding_prob)
        # print(f'group_clients_correspoding_prob: {group_clients_correspoding_prob}')
        # class-balanced sampling
        for i in range(len(group_clients_correspoding_prob)):
            if group_clients_correspoding_prob[i] > 0:
                group_clients_correspoding_prob[i] = 1/group_clients_correspoding_prob[i]
        normalized_group_clients_correspoding_prob = group_clients_correspoding_prob / sum(group_clients_correspoding_prob)
        # print('normalized_group_clients_correspoding_prob', normalized_group_clients_correspoding_prob)
        # for each sample
        for i in range(len(normalized_group_clients_correspoding_prob)):
            if normalized_group_clients_correspoding_prob[i]:
                normalized_group_clients_correspoding_prob[i] /= (len(self.data_split) * self.cur_client_prob_distribution[i])
        # print('dier normalized_group_clients_correspoding_prob', normalized_group_clients_correspoding_prob)
        # print('dier', normalized_group_clients_correspoding_prob)
        target_list = np.array([self.dataset[index]['target'].item() for index in self.data_split])
        reweight_local_data_prob = np.array([normalized_group_clients_correspoding_prob[target] for target in target_list])
        # reweight_local_data_prob = reweight_local_data_prob / sum(reweight_local_data_prob)
        # b = sum(reweight_local_data_prob)
        end = time.time()
        # print('haoshi:', end-start)
        return reweight_local_data_prob




    

    