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
from collections import defaultdict

from .serverBase import ClientSampler

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

from utils.api import (
    to_device,  
    collate
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


class ServerFedEnsemble(ServerBase):

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
        self.selected_client_ids = None


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
        self.selected_client_ids = selected_client_ids       
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
                client_id=client_id,
                max_local_gradient_update=None,
                high_freq_clients=None,
            )

            data_loader_list.append(make_data_loader(
                dataset={'train': dataset}, 
                tag='client',
                batch_sampler={'train': client_sampler}
            )['train'])

        for i in range(num_active_clients):
            m = selected_client_ids[i]

            self.clients[m].active = True
            self.clients[m].train(
                # dataset=dataset_m, 
                data_loader = data_loader_list[i],
                lr=lr, 
                metric=metric, 
                logger=logger,
            )

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
        global_epoch,
        # **kwargs
    ):  
        data_loader = make_data_loader(
            dataset={'test': dataset}, 
            tag='server'
        )['test']
        
        test_models = []
        for i in range(len(self.selected_client_ids)):
            m = self.selected_client_ids[i]
            # if self.clients[m].active == True:
            test_models.append(super().create_test_model(
                model_state_dict=self.clients[m].model_state_dict,
                batchnorm_dataset=batchnorm_dataset
            ))

        if len(test_models) == 0:
            raise ValueError('Ensemble test models are empty')

        logger.safe(True)
        with torch.no_grad():
            test_acc = 0
            loss = 0

            batch_count = 0
            for i, input in enumerate(data_loader):

                # a = input
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                
                # evaluate ensemble
                target_logit_output = 0
                for test_model in test_models:
                    test_model.train(False)
                    temp_input = copy.deepcopy(input)
                    output = test_model(temp_input)
                    # print(f"output[target]: {output['target']}")
                    target_logit_output += output['target']

                target_logp = F.log_softmax(target_logit_output, dim=1)

                temp_input = copy.deepcopy(input)
                test_acc += torch.sum( torch.argmax(target_logp, dim=1) == temp_input['target'] ) / temp_input['target'].shape[0] * 100 #(torch.sum().item()
                # print(f'test_acc: {test_acc}')
                nll_loss = nn.NLLLoss()
                loss += nll_loss(target_logp, temp_input['target'])
                # print(f'loss: {loss}')
                batch_count += 1

            loss = loss.detach().cpu().item() / batch_count
            test_acc = test_acc.detach().cpu().item() / batch_count


            evaluation = {}
            evaluation['Loss'] = loss
            evaluation['Accuracy'] = test_acc
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
            logger.append(
                info, 
                'test', 
                mean=False
            )
            print(logger.write('test', metric.metric_name['test']), flush=True)
        logger.safe(False)
        return
