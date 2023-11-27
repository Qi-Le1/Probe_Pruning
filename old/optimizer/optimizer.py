from _typing import (
    ModelType,
    Tag,
    OptimizerType,
    SchedulerType
)

from config import cfg

import torch.optim as optim


def create_optimizer(
    model: ModelType, 
    tag: Tag,
    ratio=1
) -> OptimizerType:
    '''
    Create optimizer for current model according to cfg[tag]['optimizer_name']
    
    Parameters
    ----------
    model: ModelType
    tag: Tag

    Returns
    -------
    OptimizerType
    '''
    # print('optimizer ratio:', ratio)
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=ratio*cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                                weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                               weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer