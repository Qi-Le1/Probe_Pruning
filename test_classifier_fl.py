import argparse
import os
import re
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models
# from thop import profile
from config import cfg, process_args
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
    make_batchnorm,
    InferenceConv2d,
    InferenceLinear
)

from torchinfo import summary

# from utils import save, to_device, process_control, process_dataset, resume, collate
from utils.api import (
    save, 
    to_device, 
    process_command, 
    process_dataset,  
    resume, 
    collate
)

from models.api import (
    # CNN,
    create_model,
    make_batchnorm
)

from logger import make_logger

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
        a = '_'.join(cfg['model_tag'].split('_')[:8])
        cfg['test_load_model_tag'] = '_'.join(cfg['model_tag'].split('_')[:8])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    
    
    return



def _get_submodules(model, key):
    b = ".".join(key.split(".")[:-1])
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def _replace_module(parent_module, child_name, new_module, old_module):
        new_module.to(old_module.weight_orig.device)
        setattr(parent_module, child_name, new_module)
        return
        # new_module.weight = old_module.weight
        # if hasattr(old_module, "bias"):
        #     if old_module.bias is not None:
        #         new_module.bias = old_module.bias

        # if getattr(old_module, "state", None) is not None:
        #     new_module.state = old_module.state
        #     new_module.to(old_module.weight.device)

        # # dispatch to correct device
        # for name, module in new_module.named_modules():
        #     if "lora_" in name:
        #         module.to(old_module.weight.device)


def pre_hook_prune_weight(module, input):
    
    """Create a LayerInfo object to aggregate layer information."""
    # del inputs
    # info = LayerInfo(var_name, module, curr_depth, parent_info)
    # info.calculate_num_params()
    # info.check_recursive(layer_ids)
    # summary_list.append(info)
    # layer_ids.add(info.layer_id)
    # global_layer_info[info.layer_id] = info

    input_data = input[0]
    selected_channels = None
    if len(input) == 2:
        selected_channels = input[1]

    if selected_channels is not None:
        if module.layer_type == 'conv':
            module.conv.weight = nn.Parameter(module.conv.weight_orig[:, selected_channels, :, :])

            # Create a tensor of zeros with the same shape as the original weights
            # binary_mask = torch.zeros_like(module.conv.weight_orig)

            # # Set positions corresponding to selected_channels to one
            # binary_mask[:, selected_channels, :, :] = 1

            # # Convert the binary mask tensor to a Parameter
            # module.conv.weight_mask = binary_mask

        elif module.layer_type == 'linear':
            module.linear.weight = nn.Parameter(module.linear.weight_orig[:, selected_channels])

            # Create a tensor of zeros with the same shape as the original weights
            # binary_mask = torch.zeros_like(module.linear.weight_orig)

            # # Set positions corresponding to selected_channels to one
            # binary_mask[:, selected_channels] = 1

            # # Convert the binary mask tensor to a Parameter
            # module.linear.weight_mask = binary_mask
    # else:
    #     if module.layer_type == 'conv':
    #         # module.conv_mask.weight = nn.Parameter(module.conv_orig.weight[:, selected_channels, :, :])

    #         # Create a tensor of zeros with the same shape as the original weights
    #         binary_mask = torch.ones_like(module.conv.weight_orig)

    #         # Convert the binary mask tensor to a Parameter
    #         module.conv.weight_mask = binary_mask

    #     elif module.layer_type == 'linear':
    #         # module.linear_mask.weight = nn.Parameter(module.linear_orig.weight[:, selected_channels])

    #         # Create a tensor of zeros with the same shape as the original weights
    #         binary_mask = torch.ones_like(module.linear.weight_orig)

    #         # Set positions corresponding to selected_channels to one
    #         # binary_mask[:, selected_channels] = 1

    #         # Convert the binary mask tensor to a Parameter
    #         module.linear.weight_mask = binary_mask
    # if selected_channels:
    #     print('my pre_hook_prune_weight: ', len(selected_channels))
    # else:
    #     print('my pre_hook_prune_weight: ')
    return

def replace_module(model, logger):

    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        # if isinstance(peft_config.target_modules, str):
        # target_module_found = re.fullmatch(cfg['replace_model_config'][cfg['model_name']], key)
        # else:
        # print(f'key: {key}')
        # target_module_found = any(key.endswith(target_key) or key.startswith(target_key) for target_key in cfg['replace_model_config'][cfg['model_name']])
        target_module_found = any(target_key in key for target_key in cfg['replace_model_config'][cfg['model_name']])
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, nn.Conv2d):
                new_module = InferenceConv2d(target)
            elif isinstance(target, nn.Linear):
                new_module = InferenceLinear(target)

            new_module.register_forward_pre_hook(pre_hook_prune_weight)
            # print('target_found_name: {}'.format(target_name))
            # for name, module in new_module.named_modules():
                # print('new_module Name', name)
            # if cfg['delete_method'] == 'unstructured' or cfg['delete_method'] == 'channel-wise' or cfg['delete_method'] == 'filter-wise':
            #     evaluation = {}
            #     for attr, value in vars(new_module.parameter_deletor).items():
            #         key = f'{key}_{attr}'
            #         # evaluation[key] = value
            #         if attr == 'delete_channel_ratio':
            #             print(f'replace_module/{key}: {value}')
            #         if type(value) != str:
            #             evaluation[key] = value
            #     logger.append(evaluation, 'test', 1)
            #     logger.safe(False)
            _replace_module(parent, target_name, new_module, target)
    return


def save_intermediate_info(model, logger, evaluation, MACs_ratio):
    for name, module in model.named_modules():
        if 'relu' in name:
            for attr, value in vars(module.channel_deletor).items():
                key = f'{name}_{attr}'
                evaluation[key] = value
                # if attr == 'PQ_index':
                #     print(f'{key}: {value}')
            # key = f'name'
            # for key, val in module.layer_sparsity[module.batch_size][module.relu_threshold].items():
            #     if check_type(val):
            #         cur_key = f'{key}_{name}_{module.batch_size}_{module.relu_threshold}'
            #         evaluation[cur_key] = val

            # empty_all_channel_ratio = module.layer_sparsity[module.batch_size][module.relu_threshold]['empty_all_channel'] / module.layer_sparsity[module.batch_size][module.relu_threshold]['total_channel']
            # cur_key = f'empty_all_channel_ratio_{name}_{module.batch_size}_{module.relu_threshold}'
            # evaluation[cur_key] = empty_all_channel_ratio
            
            # logger.append(evaluation, 'test', 1)
            
            # if 'PQ_index_list_distribution_mean' in module.layer_sparsity[module.batch_size][module.relu_threshold]:
            #     evaluation = {
            #         f'PQ_index_list_distribution_mean_{name}_{module.batch_size}_{module.relu_threshold}': module.layer_sparsity[module.batch_size][module.relu_threshold]['PQ_index_list_distribution_mean']
            #     }
            #     # logger.append(evaluation, 'test', 1)
            #     logger.append(
            #         evaluation, 
            #         'test', 
            #         n = len(module.layer_sparsity[module.batch_size][module.relu_threshold]['PQ_index_list_distribution_mean'])
            #     )          
    evaluation['MACs_ratio'] = MACs_ratio
    # logger.safe(False)
    # logger.reset()
    return

def runExperiment():
    cfg['current_mode'] = 'test'
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    
    result = resume(cfg['test_load_model_tag'], load_tag='checkpoint')
    train_logger = copy.deepcopy(result['logger']) if 'logger' in result else None
    last_epoch = copy.deepcopy(result['epoch'])
    data_split = copy.deepcopy(result['data_split'])

    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    
    # server_test_batch_size_list = [100, 500, 1000, 2000]
    # relu_threshold_list = [0, 0.01, 0.03, 0.07, 0.1,0.4, 0.8]
    # for batch_size in server_test_batch_size_list:
    #     cfg['server']['batch_size']['test'] = batch_size
    # for relu_threshold in relu_threshold_list:    
    model = create_model()
    model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
    batchnorm_dataset = make_batchnorm_dataset(dataset['train'])
    metric = Metric({'test': ['Loss', 'Accuracy']})
    model.load_state_dict(result['server'].server_model_state_dict)
    replace_module(model, test_logger)
    

    data_loader = make_data_loader(dataset, 'server')
    # test_model = make_batchnorm_stats(batchnorm_dataset, model, 'server')
    test(data_loader['test'], model, metric, test_logger, last_epoch)
    # save_intermediate_info(test_model, train_logger)
        
    result = {'cfg': cfg, 
              'logger': {'train': train_logger, 'test': test_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return 1

def register_forward_hooks(model):
    for name, module in model.named_modules():
        if 'relu' in name:
            module.register_hooks(set_column_to_zero=False)
    return

def check_type(var):
    if isinstance(var, int):
        return True
    elif isinstance(var, float):
        return True
    else:
        return False
    


def test(data_loader, model, metric, logger, epoch): 
    
    # for name, module in model.named_modules():
    #     print(f'name: {name}')

    original_model_MACs = None
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            # logger.safe(True)
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            temp_input = copy.deepcopy(input)
            if original_model_MACs is None:
                delete_criteria_temp = cfg['delete_criteria']
                cfg['delete_criteria'] = 'None'
                model_stats = summary(model, input_data=[temp_input], col_names=["output_size", "num_params", "mult_adds"])
                original_model_MACs = model_stats.total_mult_adds
                cfg['delete_criteria'] = delete_criteria_temp

            output = model(input)
            # del temp_input['id']
            print('-----------\n')
            model_stats = summary(model, input_data=[temp_input], col_names=["output_size", "num_params", "mult_adds"])
            ERI_MACs = model_stats.total_mult_adds

            MACs_ratio = ERI_MACs / original_model_MACs
            print('MACs_ratio', ERI_MACs, original_model_MACs, MACs_ratio)
            # res = summary(model, input_size=(1, 3, 32, 32))
            # print('torchinfo res: ', res)
            # print('------\n')
            # key_name = f'ReLU_{relu_threshold}'
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            # accuracy = evaluation['Accuracy']
            # evaluation = {key_name: accuracy}
            save_intermediate_info(model, logger, evaluation, MACs_ratio)
            logger.append(evaluation, 'test', input_size)
            logger.safe(False)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
       
        print(logger.write('test', metric.metric_name['test']))
    
    logger.reset()
    return



if __name__ == "__main__":
    main()
