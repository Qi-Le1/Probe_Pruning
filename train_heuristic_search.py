from __future__ import annotations

import argparse
import datetime

import os
import sys
import copy
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from config import (
    cfg, 
    process_args
)

from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)

from metrics import Metric

from models.api import (
    # CNN,
    create_model,
    make_batchnorm
)

from modules.client.api import (
    ClientDynamicFL,
    ClientFedAvg,
    ClientFedGen,
    ClientFedProx,
    ClientDynamicSgd,
    ClientDynamicAvg,
    ClientScaffold
)

from modules.server.api import (
    ServerDynamicFL,
    ServerFedAvg,
    ServerFedEnsemble,
    ServerFedGen,
    ServerFedProx,
    ServerDynamicSgd,
    ServerDynamicAvg,
    ServerScaffold,
    ServerCombinationSearch
)

from utils.api import (
    save, 
    to_device, 
    process_command, 
    process_dataset,  
    resume, 
    collate
)

from models.api import create_model

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

from logger import Logger, make_logger

from optimizer.api import (
    create_optimizer,
    create_scheduler
)

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_command()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))

    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print(f"Experiment: {cfg['model_tag']}")
        runExperiment()
    return

def create_clients(
    model: ModelType, 
    data_split: dict[str, dict[int, list[int]]],
    communicationMetaData
) -> dict[int, ClientType]:
    '''
    Create corresponding server to cfg['algo_mode']
    
    Parameters
    ----------
    model: ModelType

    Returns
    -------
    ServerType
    '''
    if cfg['algo_mode'] == 'dynamicfl':
        return ClientDynamicFL.create_clients(
            model=model,
            data_split=data_split,
            communicationMetaData=communicationMetaData
        )
    elif cfg['algo_mode'] == 'fedavg':
        return ClientFedAvg.create_clients(
            model=model,
            data_split=data_split
        )
    elif cfg['algo_mode'] == 'fedensemble':
        return ClientFedAvg.create_clients(
            model=model,
            data_split=data_split
        )
    elif cfg['algo_mode'] == 'fedgen':
        return ClientFedGen.create_clients(
            model=model,
            data_split=data_split
        )
    elif cfg['algo_mode'] == 'fedprox':
        return ClientFedProx.create_clients(
            model=model,
            data_split=data_split
        )
    elif cfg['algo_mode'] == 'dynamicsgd':
        return ClientDynamicSgd.create_clients(
            model=model,
            data_split=data_split
        )
    elif cfg['algo_mode'] == 'dynamicavg':
        return ClientDynamicAvg.create_clients(
            model=model,
            data_split=data_split
        )
    elif cfg['algo_mode'] == 'scaffold':
        return ClientScaffold.create_clients(
            model=model,
            data_split=data_split
        )
    else:
        raise ValueError('wrong algo model')
    
def create_server(
    model: ModelType,
    clients: dict[int, ClientType],
    dataset: DatasetType,
    communicationMetaData
) -> ServerType:
    '''
    Create corresponding server to cfg['algo_mode']
    
    Parameters
    ----------
    model: ModelType

    Returns
    -------
    ServerType
    '''
    # if cfg['select_client_mode'] == 'nonpre':
    # communicationMetaData = ClientDynamicFL.create_communication_meta_data()
    return ServerCombinationSearch(model, clients, dataset, communicationMetaData)
    # else:
    #     raise ValueError('wrong')
    # return ServerCombinationSearch(model, clients, dataset)
    # if cfg['algo_mode'] == 'dynamicfl':
    #     if cfg['select_client_mode'] == 'nonpre':
    #         communicationMetaData = ClientDynamicFL.create_communication_meta_data()
    #         return ServerDynamicFL(model, clients, dataset, communicationMetaData)
    #     elif cfg['select_client_mode'] == 'fix':
    #         return ServerDynamicFL(model, clients, dataset)
    #     else:
    #         raise ValueError('wrong select client mode for dynamicfl')
    # elif cfg['algo_mode'] == 'fedavg':
    #     return ServerFedAvg(model, clients, dataset)
    # elif cfg['algo_mode'] == 'fedensemble':
    #     return ServerFedEnsemble(model, clients, dataset)
    # elif cfg['algo_mode'] == 'fedgen':
    #     return ServerFedGen(model, clients, dataset)
    # elif cfg['algo_mode'] == 'fedprox':
    #     return ServerFedProx(model, clients, dataset)
    # elif cfg['algo_mode'] == 'dynamicsgd':
    #     return ServerDynamicSgd(model, clients, dataset)
    # elif cfg['algo_mode'] == 'dynamicavg':
    #     return ServerDynamicAvg(model, clients, dataset)
    # elif cfg['algo_mode'] == 'scaffold':
    #     return ServerScaffold(model, clients, dataset)
    # else:
    #     raise ValueError('wrong algo model')



def runExperiment():
    global cfg
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    # torch.set_default_dtype(torch.float64)
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    train_data_num = len(dataset['train'])
    if cfg['max_local_gradient_update'] == 'no_input':
        cfg['max_local_gradient_update'] = int(train_data_num / cfg['num_clients'] \
        * cfg['local_epoch'] / cfg['client']['batch_size']['train'])
    print(f'max_local_gradient_update: {cfg["max_local_gradient_update"]}')
    # print('\n ****')
    # cur_id = 0
    # print(f'cur_id: {cur_id}')
    # temp_list_2 = []
    # for vector in dataset['train'][cur_id]['data']:
    #     for sub in vector:
    #         # print(f'vector: {vector} \n')
    #         temp = copy.deepcopy(sub)
    #         temp = temp.tolist()
    #         temp.sort()
    #         temp_list_2.append(copy.deepcopy(temp))
    # temp_list_2.sort()
    # print(f'length: {len(temp_list_2)}')
    # for i in range(len(temp_list_2)):
    #     print(f'{i}: {temp_list_2[i]} \n') 
    # print('**** \n')

    # print('\n ****')
    # cur_id = 0
    # print(f'cur_id: {cur_id}')
    # temp_list_3 = []
    # for vector in dataset['train'][cur_id]['data']:
    #     for sub in vector:
    #         # print(f'vector: {vector} \n')
    #         temp = copy.deepcopy(sub)
    #         temp = temp.tolist()
    #         temp.sort()
    #         temp_list_3.append(copy.deepcopy(temp))
    # temp_list_3.sort()
    # print(f'length: {len(temp_list_3)}')
    # for i in range(len(temp_list_3)):
    #     print(f'{i}: {temp_list_3[i]} \n') 
    # print('**** \n')

    process_dataset(dataset)
    # data_loader = make_data_loader(dataset, 'global')
    model = create_model()
    optimizer = create_optimizer(model, 'client')
    scheduler = create_scheduler(optimizer, 'server')
    batchnorm_dataset = make_batchnorm_dataset(dataset['train'])
    data_split = split_dataset(dataset, cfg['num_clients'], cfg['data_split_mode'])
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    result = resume(cfg['model_tag'], resume_mode=cfg['resume_mode'])
    if result is None:
        last_global_epoch = 1
        communicationMetaData = None
        if cfg['algo_mode'] == 'dynamicfl':
            client_ids = torch.arange(cfg['num_clients'])
            communicationMetaData = ClientDynamicFL.create_communication_meta_data(client_ids=client_ids)
        clients = create_clients(
            model=model, 
            data_split=data_split,
            communicationMetaData=communicationMetaData
        )
        server = create_server(
            model=model, 
            clients=clients, 
            dataset=copy.deepcopy(dataset['train']),
            communicationMetaData=communicationMetaData
        )
        if cfg['algo_mode'] == 'fedgen':
            for client in clients:
                client.generative_model = server.generative_model
                
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        cfg = result['cfg']
        last_global_epoch = result['epoch']
        server = result['server']
        clients = result['clients']
        optimizer.load_state_dict(result['optimizer_state_dict'])
        scheduler.load_state_dict(result['scheduler_state_dict'])
        data_split = result['data_split']
        logger = result['logger']
    
    cfg['server']['num_epochs'] = 100
    for global_epoch in range(last_global_epoch, cfg['server']['num_epochs'] + 1):
        start = time.time()
        server.train(
            dataset=copy.deepcopy(dataset['train']), 
            optimizer=optimizer, 
            metric=metric, 
            logger=logger, 
            global_epoch=global_epoch
        )
        end = time.time()
        print(f"{cfg['model_tag']}, round: {global_epoch}-----\n")
        print(f'time cost: {end-start}-----\n')
        # scheduler.step()
        # server.evaluate_trained_model(
        #     dataset=copy.deepcopy(dataset['test']),
        #     logger=logger,
        #     metric=metric,
        #     global_epoch=global_epoch
        # )
        

        result = {
            'cfg': cfg, 
            'epoch': global_epoch + 1, 
            # 'server': server, 
            # 'clients': clients,
            # 'server_model_state_dict': server.server_model_state_dict,
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(), 
            # 'data_split': data_split, 
            'logger': logger,
        }

        # print(f'size of server: {sys.getsizeof(server)}')
        # print(f'size of clients: {sys.getsizeof(clients)}')
        # print(f'size of optimizer: {sys.getsizeof(optimizer)}')
        # print(f'size of scheduler: {sys.getsizeof(scheduler)}')
        # print(f'size of logger: {sys.getsizeof(logger)}')
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        # if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
        #     metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
        # shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
        #             './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    
    
    return

if __name__ == "__main__":
    main()
