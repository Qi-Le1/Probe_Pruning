from numpy import number
from config import cfg
from typing import List
import collections



def process_command():
    if 'relu_threshold' in cfg['control']:
        cfg['relu_threshold'] = float(cfg['control']['relu_threshold'])
    else:
        cfg['relu_threshold'] = 0
        
    cfg['algo_mode'] = cfg['control']['algo_mode']
        
    if 'local_epoch' in cfg['control']:
        cfg['local_epoch'] = int(cfg['control']['local_epoch'])
    cfg['save_interval'] = 50
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['algo_mode'] = cfg['control']['algo_mode']
    cfg['merge_gap'] = False
    # if 'merge_gap' in cfg['control']:
    #     cfg['merge_gap'] = True if int(cfg['control']['merge_gap']) == 1 else False
    # if 'local_upper_bound' in cfg['control']:
    #     cfg['local_upper_bound'] = int(cfg['control']['local_upper_bound'])
    # else:
    #     cfg['local_upper_bound'] = 99999
    if 'data_prep_norm_test' in cfg['control']:
        cfg['data_prep_norm_test'] = cfg['control']['data_prep_norm_test']
    else:
        cfg['data_prep_norm_test'] = 'bn'
    
    if 'norm' in cfg['control']:
        cfg['norm'] = cfg['control']['norm']
    else:
        cfg['norm'] = 'ln'
    # if 'grad' in cfg['control']:
    #     cfg['grad'] = cfg['control']['grad']
    # else:
    #     cfg['grad'] = 'noclip'
    if 'server_aggregation' in cfg['control']:
        cfg['server_aggregation'] = cfg['control']['server_aggregation']
    else:
        cfg['server_aggregation'] = 'WA'
    
    if 'no_training' in cfg['control']:
        cfg['no_training'] = cfg['control']['no_training']

    if 'data_prep_norm' in cfg['control']:
        cfg['data_prep_norm'] = cfg['control']['data_prep_norm']
    else:
        cfg['data_prep_norm'] = 'bn'


    cfg['dp_ensemble_times'] = 10

    data_shape = {'CIFAR10': [3, 32, 32], 'CIFAR100': [3, 32, 32], 'SVHN': [3, 32, 32], 'MNIST': [1, 28, 28], 'FEMNIST': [1, 28, 28]}
    cfg['data_shape'] = data_shape[cfg['data_name']]
    cfg['conv'] = {'hidden_size': [32, 64]}
    cfg['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    # cfg['resnet9'] = {'hidden_size': [64, 128]}
    cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    cfg['change_batch_size'] = False
    if 'change_batch_size' in cfg['control']:
        cfg['change_batch_size'] = True if int(cfg['control']['change_batch_size']) == 1 else False
    cfg['change_lr'] = False
    if 'change_lr' in cfg['control']:
        cfg['change_lr'] = True if int(cfg['control']['change_lr']) == 1 else False
    if 'reweight_sample' in cfg['control']:
        cfg['reweight_sample'] = True if int(cfg['control']['reweight_sample']) == 1 else False
    
    cfg['cal_communication_cost'] = False
    if 'cal_communication_cost' in cfg['control']:
        cfg['cal_communication_cost'] = True if int(cfg['control']['cal_communication_cost']) == 1 else False
    
    cfg['only_high_freq'] = False
    if 'only_high_freq' in cfg['control']:
        cfg['only_high_freq'] = True if int(cfg['control']['only_high_freq']) == 1 else False
    # cfg['batch_size_threshold'] = 5
    cfg['threshold'] = 0.95
    cfg['alpha'] = 0.75
    cfg['feddyn_alpha'] = 0.1
    cfg['max_clip_norm'] = 10
    if 'num_clients' in cfg['control']:
        cfg['num_clients'] = int(cfg['control']['num_clients'])
        cfg['active_rate'] = float(cfg['control']['active_rate'])
        # cfg['active_rate'] = 0.1
        cfg['data_split_mode'] = cfg['control']['data_split_mode']
        # cfg['diff_val'] = float(cfg['control']['diff_val'])
        
        cfg['gm'] = 0
        cfg['server'] = {}
        cfg['server']['shuffle'] = {'train': True, 'test': False}
        
        cfg['server']['batch_size'] = {'train': 250, 'test': 500}
        if 'test_server_batch_size' in cfg['control']:
            cfg['server']['batch_size']['test'] = int(cfg['control']['test_server_batch_size'])
        cfg['client'] = {}
        cfg['client']['shuffle'] = {'train': True, 'test': False}
        cfg['client']['batch_size'] = {'train': 10, 'test': 500}
        # cfg['client']['batch_size'] = {'train': 16, 'test': 500}
        if 'train_batch_size' in cfg['control']:
            cfg['client']['batch_size']['train'] = int(cfg['control']['train_batch_size'])


        if 'relu_threshold' in cfg['control']:
            cfg['relu_threshold'] = float(cfg['control']['relu_threshold'])

        if 'delete_criteria' in cfg['control']:
            cfg['delete_criteria'] = cfg['control']['delete_criteria']
        
        if 'delete_threshold' in cfg['control']:
            cfg['delete_threshold'] = float(cfg['control']['delete_threshold'])
        cfg['normalized_model_size'] = 1
        cfg['client']['optimizer_name'] = 'SGD'
        cfg['client']['lr'] = 3e-2
        if cfg['model_name'] == 'cnn':
            cfg['client']['lr'] = 1e-2
            # cfg['client']['lr'] = 5e-3
        cfg['client']['momentum'] = 0.9
        cfg['client']['weight_decay'] = 5e-4
        cfg['client']['nesterov'] = True

        # cfg['client']['lr'] = 3e-2
        # cfg['client']['momentum'] = cfg['gm']
        # cfg['client']['weight_decay'] = 0
        # cfg['client']['nesterov'] = False

        cfg['client']['num_epochs'] = cfg['local_epoch']

        # if cfg['num_clients'] > 10:
        #     cfg['server']['num_epochs'] = 5
        # else:

        # for grouping high freq clients
        # if len(cfg['control']) == 9:
            # cfg['server']['num_epochs'] = 5
        # for training the experiments
        # elif len(cfg['control']) == 10:
        cfg['server']['num_epochs'] = 800
        # cfg['server']['num_epochs'] = 5
        cfg['server']['optimizer_name'] = 'SGD'
        cfg['server']['lr'] = 1
        cfg['server']['momentum'] = cfg['gm']
        cfg['server']['weight_decay'] = 0
        cfg['server']['nesterov'] = False
        cfg['server']['scheduler_name'] = 'CosineAnnealingLR'

        cfg['prune_norm'] = 2
        if 'prune_norm' in cfg['control']:
            cfg['prune_norm'] = int(cfg['control']['prune_norm'])
        
        cfg['delete_method'] = 'our'
        if 'delete_method' in cfg['control']:
            cfg['delete_method'] = cfg['control']['delete_method']
        
        if 'batch_deletion' in cfg['control']:
            cfg['batch_deletion'] = cfg['control']['batch_deletion']
        else:
            cfg['batch_deletion'] = 'PQ'

    else:
        raise ValueError('no num_clients')

    
    cfg['upload_freq_level'] = {
        6: 1,
        5: 4,
        4: 16,
        3: 32,
        2.5: 64,
        2: 128,
        1: 256
    }

    cfg['replace_model_config'] = {
        'cnn': ['conv', 'fc'],
        'resnet18': ['conv', 'shortcut', 'linear'],
        
    }

    print(f'cfg: {cfg}', flush=True)
    return
