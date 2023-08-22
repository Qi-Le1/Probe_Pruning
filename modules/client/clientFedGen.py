from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import torch
import collections
import torch.nn.functional as F
import models
import torch.nn as nn
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

from utils.api import (
    CONFIGS_,
    RUNCONFIGS
)

class ClientFedGen(ClientBase):

    def __init__(
        self, 
        client_id: int, 
        model: ModelType, 
        data_split: list[int],
        available_labels
    ) -> None:

        super().__init__()
        self.init_loss_fn()
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = create_optimizer(model, 'client')
        self.optimizer_state_dict = optimizer.state_dict()
        self.active = False

        dataset_name = cfg['data_name']
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']

        # batch_size and gen_batch_size are equal in FedGen
        self.batch_size = cfg['client']['batch_size']['train']
        self.gen_batch_size = cfg['client']['batch_size']['train']
        self.latent_layer_idx = -1

        self.available_labels = available_labels
        self.label_counts = collections.defaultdict(int)
        # self.label_info = label_info

    def init_loss_fn(self):
        # negative log likelihood loss
        self.nll_loss=nn.NLLLoss()
        self.mse_loss = nn.MSELoss()
        self.kldiv_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()


    @classmethod
    def create_clients(
        cls,
        model: ModelType, 
        data_split: dict[str, dict[int, list[int]]],
        dataset
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
            available_labels = np.array([dataset[index]['target'].item() for index in data_split['train'][m]])
            clients[m] = ClientFedGen(
                client_id=client_id[m], 
                model=model, 
                data_split={
                    'train': data_split['train'][m], 
                    'test': data_split['test'][m]
                },
                available_labels=available_labels
            )
        return clients

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, target):
        unique_y, counts=torch.unique(target, return_counts=True)
        unique_y = unique_y.detach().numpy()
        counts = counts.detach().numpy()
        for label, count in zip(unique_y, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:0 for label in range(self.unique_labels)}

    def train(
        self, 
        # dataset: DatasetType, 
        data_loader,
        lr: int, 
        metric: MetricType, 
        logger: LoggerType,
        global_epoch: int,
        generative_model
    ) -> None:

        model = create_model()
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = create_optimizer(model, 'client')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        generative_model.train(False)

        # self.clean_up_counts()
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        for i, input in enumerate(data_loader):

            input = collate(input)
            self.update_label_counts(target=input['target'])            
            input_size = input['data'].size(0)                
            input = to_device(input, cfg['device'])
            optimizer.zero_grad()
            temp_input = copy.deepcopy(input)
            output = model(input)
            user_output_logp = F.log_softmax(output['target'], dim=1)
            # cross-entropy loss
            predictive_loss = output['loss']

            generative_alpha = self.exp_lr_scheduler(global_epoch, decay=0.98, init_lr=self.generative_alpha)
            generative_beta = self.exp_lr_scheduler(global_epoch, decay=0.98, init_lr=self.generative_beta)
            ### get generator output(latent representation) of the same label
            target = temp_input['target']
            gen_output = generative_model(target, latent_layer_idx=self.latent_layer_idx)['output']

            logit_given_gen = model({'data': gen_output}, start_layer_idx=self.latent_layer_idx)['target']
            target_p = F.softmax(logit_given_gen, dim=1).clone().detach()

            user_latent_loss = generative_beta * self.kldiv_loss(user_output_logp, target_p)

            sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)
            sampled_y = to_device(torch.tensor(sampled_y).type(torch.int64), cfg['device'])
            gen_result = generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)
            gen_output = gen_result['output'] # latent representation when latent = True, x otherwise
            user_output_target = model({'data': gen_output}, start_layer_idx=self.latent_layer_idx)['target']
            user_output_logp = F.log_softmax(user_output_target, dim=1)

            teacher_loss = generative_alpha * torch.mean(
                generative_model.crossentropy_loss(user_output_logp, sampled_y)
            )
            # this is to further balance oversampled down-sampled synthetic data
            gen_ratio = self.gen_batch_size / self.batch_size
            loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss
            TEACHER_LOSS += teacher_loss
            LATENT_LOSS += user_latent_loss

            loss.backward()
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
                
        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return