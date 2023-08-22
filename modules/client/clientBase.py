from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import math
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg

# from torchstat import stat

from utils.api import (
    to_device,  
    collate
)

from models.api import (
    create_model,
    make_batchnorm
)

from _typing import (
    ClientType,
    ModelType,
    DatasetType,
    MetricType,
    LoggerType
)

from models.api import make_batchnorm

from optimizer.api import create_optimizer

from data import make_data_loader


class ClientBase:

    def __init__(self) -> None:
        # self.init_loss_fn()
        pass
    
    def reform_model_output(self, output, loss):
        '''
        Reform the structure of output to adapt the original code
        with FedGen / FedEnsemble
        '''
        res = {
            'target': output,
            'loss': loss
        }
        return res


    def update_optimizer_state_dict(
        self,
        client_model_state_dict,
        client_optimizer_state_dict,
        client_optimizer_lr,
        server_model_state_dict
    ):
        '''
        Mitigate the gap between the optimizer state dict for 
        client model and the new server model
        '''
        client_model = create_model(track_running_stats=False, on_cpu=True)
        client_model.load_state_dict(client_model_state_dict, strict=False)
        client_optimizer_state_dict['param_groups'][0]['lr'] = client_optimizer_lr
        client_optimizer = create_optimizer(client_model, 'client')
        client_optimizer.load_state_dict(client_optimizer_state_dict)
        with torch.no_grad():  
            for k, v in client_model.named_parameters():
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    tmp_v = v.data.new_zeros(v.size())
                    tmp_v += server_model_state_dict[k]
                    v.grad = (v.data - tmp_v).detach()
            # clip
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), 1)
            client_optimizer.step()
        return copy.deepcopy(client_optimizer.state_dict())


