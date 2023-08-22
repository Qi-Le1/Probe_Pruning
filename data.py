from __future__ import annotations

import copy
import torch
import numpy as np
# import models
import datasets
from torch import nn
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from utils.api import (
    collate, 
    to_device
)

from _typing import (
    DatasetType
)

from models.api import make_batchnorm

data_stats = {
    'MNIST': ((0.1307,), (0.3081,)), 
    #TODO
    'FEMNIST': ((0.9614,), (0.1634,)), 
    'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
}

data_dimension_stats = {
    'MNIST': [1, 28, 28], 
    #TODO
    'FEMNIST': [1, 28, 28],  
    'CIFAR10': [3, 32, 32], 
    'CIFAR100': [3, 32, 32], 

}

def fetch_dataset(data_name):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['MNIST', 'FEMNIST']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        if cfg['data_prep_norm'] == 'bn':
            dataset['train'].transform = datasets.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])])
        elif cfg['data_prep_norm'] == 'ln':
            dataset['train'].transform = datasets.Compose([
                transforms.ToTensor(),
                nn.LayerNorm(data_dimension_stats[data_name])])
        
        
        if cfg['data_prep_norm'] == 'ln' and cfg['data_prep_norm_test'] == 'ln':
            dataset['test'].transform = datasets.Compose([
                transforms.ToTensor(),
                nn.LayerNorm(data_dimension_stats[data_name])])
        else:
            dataset['test'].transform = datasets.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        
        if cfg['data_prep_norm'] == 'bn':
            dataset['train'].transform = datasets.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])])
        elif cfg['data_prep_norm'] == 'ln':
            dataset['train'].transform = datasets.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                nn.LayerNorm(data_dimension_stats[data_name])])
        
        if cfg['data_prep_norm'] == 'ln' and cfg['data_prep_norm_test'] == 'ln':
            dataset['test'].transform = datasets.Compose([
                transforms.ToTensor(),
                nn.LayerNorm(data_dimension_stats[data_name])])
        else:
            dataset['test'].transform = datasets.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


def input_collate(batch):
    a = batch
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        b = output
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None, batch_sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is not None:
            data_loader[k] = DataLoader(
                dataset=dataset[k], 
                batch_size=_batch_size, 
                sampler=sampler[k],
                pin_memory=cfg['pin_memory'], 
                num_workers=cfg['num_workers'],
                collate_fn=input_collate, 
                worker_init_fn=np.random.seed(cfg['seed'])
            )
        elif batch_sampler is not None:
            data_loader[k] = DataLoader(
                dataset=dataset[k], 
                batch_sampler=batch_sampler[k],
                pin_memory=cfg['pin_memory'], 
                num_workers=cfg['num_workers'],
                collate_fn=input_collate, 
                worker_init_fn=np.random.seed(cfg['seed'])
            )
        else:
            data_loader[k] = DataLoader(
                dataset=dataset[k], 
                batch_size=_batch_size, 
                shuffle=_shuffle,
                pin_memory=cfg['pin_memory'], 
                num_workers=cfg['num_workers'],
                collate_fn=input_collate, 
                worker_init_fn=np.random.seed(cfg['seed'])
            )

    return data_loader


def split_dataset(
    dataset: dict[str, DatasetType], 
    num_users: int, 
    data_split_mode: str
) -> dict[str, dict[int, list[int]]]:

    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'] = iid(dataset['train'], num_users)
        data_split['test'] = iid(dataset['test'], num_users)
    elif 'non-iid' in cfg['data_split_mode']:
        data_split['train'] = non_iid(dataset['train'], num_users)
        data_split['test'] = non_iid(dataset['test'], num_users)
    else:
        raise ValueError('Not valid data split mode')
    # print('data-split', data_split)
    return data_split


def iid(
    dataset: DatasetType, 
    num_users: int
) -> dict[int, list[int]]:

    length = len(dataset)
    num_items = int(length / num_users)
    data_split, idx = {}, list(range(length))
    # print(f'iid data split: num_users: {num_users}, num_items: {num_items}')
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split


def non_iid(
    dataset: DatasetType, 
    num_users: int
) -> dict[int, list[int]]:

    target = torch.tensor(dataset.target)
    data_split_mode_list = cfg['data_split_mode'].split('-')
    data_split_mode_tag = data_split_mode_list[-2]
    if data_split_mode_tag == 'l':
        # 只有某n类数据
        data_split = {i: [] for i in range(num_users)}
        shard_per_user = int(data_split_mode_list[-1])
        target_idx_split = {}
        shard_per_class = int(shard_per_user * num_users / cfg['target_size'])
        for target_i in range(cfg['target_size']):
            target_idx = torch.where(target == target_i)[0]
            num_leftover = len(target_idx) % shard_per_class
            leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
            new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
            new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
            for i, leftover_target_idx in enumerate(leftover):
                new_target_idx[i] = new_target_idx[i] + [leftover_target_idx.item()]
            target_idx_split[target_i] = new_target_idx
        target_split = list(range(cfg['target_size'])) * shard_per_class
        target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
        target_split = torch.tensor(target_split).reshape((num_users, -1)).tolist()
        for i in range(num_users):
            for target_i in target_split[i]:
                idx = torch.randint(len(target_idx_split[target_i]), (1,)).item()
                data_split[i].extend(target_idx_split[target_i].pop(idx))
    elif data_split_mode_tag == 'd':
        # Dirichlet distributon
        beta = float(data_split_mode_list[-1])
        dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_users))
        min_size = 0
        required_min_size = 10
        N = target.size(0)
        proportions = dir.sample()
        while min_size < required_min_size:
            # print(f'min_size: {min_size}')
            data_split = [[] for _ in range(num_users)]
            for target_i in range(cfg['target_size']):
                target_idx = torch.where(target == target_i)[0]
                proportions = dir.sample()
                proportions = torch.tensor(
                    [p * (len(data_split_idx) < (N / num_users)) for p, data_split_idx in zip(proportions, data_split)])
                proportions = proportions / proportions.sum()
                # torch.cumsum累加最后一个纬度去分index
                a = (torch.cumsum(proportions, dim=-1) * len(target_idx))
                b = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long()
                split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                split_idx = torch.tensor_split(target_idx, split_idx)
                data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]
            min_size = min([len(data_split_idx) for data_split_idx in data_split])
        data_split = {i: data_split[i] for i in range(num_users)}

        total = []
        for key,val in data_split.items():
            total.append(len(val))
            
    else:
        raise ValueError('Not valid data split mode tag')
    
    length_list = []
    for key, item in data_split.items():
        length_list.append(len(item))
    print('length_list', length_list)
    sum_150 = sum(i<150 for i in length_list)
    print(sum_150)
    sum_250 = sum(i<250 for i in length_list)
    print(sum_250)
    sum_500 = sum(i<500 for i in length_list)
    print(sum_500)
    return data_split


def separate_dataset(
    dataset: DatasetType, 
    idx: list[int]
) -> DatasetType:
    # print(f'separate_dataset idx: {idx}')
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[s] for s in idx]
    separated_dataset.target = [dataset.target[s] for s in idx]
    separated_dataset.id = list(range(len(separated_dataset.data)))
    return separated_dataset


def make_batchnorm_dataset(dataset):
    dataset = copy.deepcopy(dataset)
    plain_transform = datasets.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*data_stats[cfg['data_name']])
    ])
    dataset.transform = plain_transform
    return dataset


def make_batchnorm_stats(dataset, model, tag):
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=True))
        data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
    return test_model
