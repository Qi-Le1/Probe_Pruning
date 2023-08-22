from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import torch
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

from models.api import (
    create_model
)

from models.api import create_generative_model

from utils.api import (
    CONFIGS_,
    RUNCONFIGS,
    to_device
)

MIN_SAMPLES_PER_LABEL = 1

class ServerFedGen(ServerBase):

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

        # stop using generated samples after 20 local epochs
        # hyperparameter set by FedGen
        self.early_stop = 20  
        self.batch_size = cfg['client']['batch_size']['train']
        # self.student_model = copy.deepcopy(self.model)
        # create generative model
        self.generative_model = create_generative_model(dataset_name=cfg['data_name'], embedding=False)
        self.latent_layer_idx = self.generative_model.latent_layer_idx

        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta, self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))

        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False
        )
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98
        )

    def init_ensemble_configs(self):
        #### used for ensemble learning ####
        dataset_name = cfg['data_name']
        self.ensemble_lr = RUNCONFIGS[dataset_name].get('ensemble_lr', 1e-4)
        # self.ensemble_batch_size = RUNCONFIGS[dataset_name].get('ensemble_batch_size', 128)
        self.ensemble_epochs = RUNCONFIGS[dataset_name]['ensemble_epochs']
        self.num_pretrain_iters = RUNCONFIGS[dataset_name]['num_pretrain_iters']
        self.temperature = RUNCONFIGS[dataset_name].get('temperature', 1)
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.ensemble_alpha = RUNCONFIGS[dataset_name].get('ensemble_alpha', 1)
        self.ensemble_beta = RUNCONFIGS[dataset_name].get('ensemble_beta', 0)
        self.ensemble_eta = RUNCONFIGS[dataset_name].get('ensemble_eta', 1)
        self.weight_decay = RUNCONFIGS[dataset_name].get('weight_decay', 0)
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
        self.ensemble_train_loss = []
        self.n_teacher_iters = 5
        self.n_student_iters = 1
        print("ensemble_lr: {}".format(self.ensemble_lr) )
        # print("ensemble_batch_size: {}".format(self.ensemble_batch_size) )
        print("unique_labels: {}".format(self.unique_labels) )

    def get_label_weights(self, selected_client_ids):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for i in range(len(selected_client_ids)):
                m = selected_client_ids[i]
                weights.append(self.clients[m].label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / (np.sum(weights) + 1e-8) )
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def train_generator(self, batch_size, selected_client_ids, epoches=1, latent_layer_idx=-1, verbose=False):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return anything.
        """
        #self.generative_regularizer.train()
        self.label_weights, self.qualified_labels = self.get_label_weights(selected_client_ids)
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0

        # update_generator_(self.n_teacher_iters, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        def update_generator_(n_iters, TEACHER_LOSS, DIVERSITY_LOSS):
            self.generative_model.train(True)
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y = np.random.choice(self.qualified_labels, batch_size)
                # y_input = torch.LongTensor(y)
                y_input = to_device(torch.LongTensor(y).type(torch.int64), cfg['device'])

                ## feed to generator
                gen_result = self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                gen_output, eps = gen_result['output'], gen_result['eps']
                ##### get losses ####
                # decoded = self.generative_regularizer(gen_output)
                # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                diversity_loss = self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                ######### get teacher loss ############
                teacher_loss = 0
                teacher_logit = 0
                for idx in range(len(selected_client_ids)):
                    client = self.clients[selected_client_ids[idx]]
                # for user_idx, user in enumerate(self.selected_users):
                    client_model = create_model(track_running_stats=False, on_cpu=False)
                    client_model.load_state_dict(client.model_state_dict, strict=False)
                    client_model.train(False)
                    weight = self.label_weights[y][:, idx].reshape(-1, 1)
                    expand_weight = np.tile(weight, (1, self.unique_labels))
                    user_result_given_gen = client_model({'data': gen_output}, start_layer_idx=latent_layer_idx)['target']
                    user_output_logp_ = F.log_softmax(user_result_given_gen, dim=1)

                    teacher_loss_ = torch.mean( \
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                        to_device(torch.tensor(weight, dtype=torch.float32), cfg['device'])
                    )
                    teacher_loss += teacher_loss_
                    # teacher_logit += user_result_given_gen * torch.tensor(expand_weight, dtype=torch.float32)

                loss = self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss #(torch.mean(TEACHER_LOSS.double())).item()
                # STUDENT_LOSS += self.ensemble_beta * student_loss #(torch.mean(student_loss.double())).item()
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss #(torch.mean(diversity_loss.double())).item()
            return TEACHER_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, DIVERSITY_LOSS = update_generator_(
                self.n_teacher_iters, TEACHER_LOSS, DIVERSITY_LOSS
            )

        TEACHER_LOSS = TEACHER_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        # STUDENT_LOSS = STUDENT_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        info = "Generator: Teacher Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, DIVERSITY_LOSS)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()


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
            # dataset_m = separate_dataset(dataset, self.clients[client_id].data_split['train'])
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
                data_loader=data_loader_list[i],
                lr=lr, 
                metric=metric, 
                logger=logger,
                global_epoch=global_epoch,
                generative_model=copy.deepcopy(self.generative_model)
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

        self.train_generator(
            batch_size=self.batch_size,
            selected_client_ids=selected_client_ids,
            epoches=self.ensemble_epochs // self.n_teacher_iters,
            latent_layer_idx=self.latent_layer_idx,
            verbose=True,
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