import os
import itertools
import json
import copy

# env_dist = os.environ
# for env in env_dist:
#     print(env, env_dist[env])

# import sys
# print(sys.executable)
# print(sys.version)
# print('---')
# print(sys)

# print(sys.path)
# CUR_FILE_PATH = os.path.abspath(__file__)
# UPPER_LEVEL_PATH = os.path.dirname(CUR_FILE_PATH)
# UPPER_UPPER_LEVEL_PATH = os.path.dirname(UPPER_LEVEL_PATH)
# # TOP_LEVEL_PATH = os.path.dirname(UPPER_UPPER_LEVEL_PATH)
# print(f'Colda Test Upper Level Path Init: {UPPER_LEVEL_PATH}')
# print(f'Colda Test Upper Upper Level Path Init: {UPPER_UPPER_LEVEL_PATH}')
# # sys.path.append(UPPER_LEVEL_PATH)
# sys.path.append(UPPER_UPPER_LEVEL_PATH)
import numpy as np
import pandas as pd
from utils.api import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import random
import collections


os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='analyze_data')
parser.add_argument('--type', default='dp', type=str)
args = vars(parser.parse_args())

save_format = 'png'
result_path = './output/result'
vis_path = './output/vis/{}'.format(save_format)

num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]


def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    # controls = [exp] + data_names + model_names + [control_names]
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(file):
    controls = []

    if file == 'cnn':

        pass
    elif file == 'baseline_batch_1':
        control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
                                ['fedavg'], ['5'], ['0'], ['1'], ['1', '2'], ['our', 'unstructured', 'channel-wise', 'filter-wise'], ['PQ'], ['0', '0.001', '0.01', '0.03', '0.06', '0.1', '0.5', '1.0', '999']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)
        # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn', 'resnet18'], ['0.1'], ['100'], ['non-iid-d-0.3', 'non-iid-d-0.1', 'iid'], 
        #                         ['fedavg'], ['5'], ['0'], ['100'], ['sparsity', 'PQ'], ['0.1']]]
    elif file == 'baseline_batch_multiple':

        # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
        #                         ['fedavg'], ['5'], ['0'], ['1', '10', '100', '1000'], ['1', '2'], ['PQ'],  ['0', '0.001', '0.01', '0.03', '0.06', '0.1', '0.5', '1.0', '999'], ['PQ', 'inter']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
        #                         ['fedavg'], ['5'], ['0'], ['1', '10', '100', '1000'], ['1', '2'], ['PQ'],  ['0.001'], ['PQ', 'inter']]]
        # control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['iid'], 
        #                         ['fedavg'], ['5'], ['0'], ['10'], ['2'], ['PQ'],  ['0.001'], ['PQ', 'inter']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)



        # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
        #                         ['fedavg'], ['5'], ['0'], ['10', '100', '1000'], ['1', '2'], ['PQ'],  ['0'], ['union']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
        #                         ['fedavg'], ['5'], ['0'], ['10', '100', '1000'], ['1', '2'], ['PQ'],  ['0', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '999'], ['union']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
                                ['fedavg'], ['5'], ['0'], ['10', '100', '1000'], ['1', '2'], ['PQ'],  ['0', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '999'], ['PQ', 'inter', 'union']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)



        # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
        #                         ['fedavg'], ['5'], ['0'], ['10'], ['2'], ['PQ'],  ['0.3', '0.4', '0.5', '0.6', '0.7', '999'], ['PQ', 'inter']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
        #                         ['fedavg'], ['5'], ['0'], ['1'], ['1'], ['PQ'],  ['0', '0.001'], ['PQ', 'inter']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)
    elif file == 'observe_pq':
        control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
                                ['fedavg'], ['5'], ['0'], ['10'], ['1', '2'], ['PQ'],  ['0', '0.05', '0.07', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '999'], ['PQ', 'inter']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)
    
    return controls


def main():
    # files = ['fs', 'ps', 'cd', 'ub', 'loss', 'local-epoch', 'gm', 'sbn', 'alternate', 'fl', 'fsgd', 'frgd', 'fmatch']
    global result_path, vis_path, num_experiments, exp

    print(f"type: {args['type']}")    
    result_path = './output/result/{}'.format(args['type'])
    vis_path = './output/vis/{}'.format(args['type'])
    files = [args['type']]

    if args['type'] == 'dp' or args['type'] == 'high_freq' or args['type'] == 'new_dp':
        num_experiments = 1
    elif args['type'] == 'cnn' or args['type'] == 'resnet18' or args['type'] == 'cnn_all':
        num_experiments = 1
    elif args['type'] == 'cnn_maoge' or args['type'] == 'resnet18_maoge' or args['type'] == 'resnet18_all' or args['type'] == 'resnet18_maoge_chongfu':
        num_experiments = 1
    elif args['type'] == 'freq_ablation':
        num_experiments = 1
    elif args['type'] == 'communication_cost':
        num_experiments = 1
    elif args['type'] == 'baseline_batch_1':
        num_experiments = 1
    elif args['type'] == 'baseline_batch_multiple':
        num_experiments = 1
    elif args['type'] == 'observe_pq':
        num_experiments = 1
    else:
        raise ValueError('Not valid type')
    exp = [str(x) for x in list(range(num_experiments))]

    controls = []
    for file in files:
        controls += make_control_list(file)
    processed_result_exp, processed_result_history = process_result(controls)
    with open('{}/processed_result_exp.json'.format(result_path), 'w') as fp:
        json.dump(processed_result_exp, fp, indent=2)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    # if processed_result_exp:
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    # if processed_result_history:
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    # print(f'extracted_processed_result_history: {extracted_processed_result_history}')
    if extracted_processed_result_exp:
        df_exp = make_df_exp(extracted_processed_result_exp)
    if extracted_processed_result_history:
        df_history = make_df_history(extracted_processed_result_history)
    df_exp = {}
    make_vis(df_exp, df_history)
    return


def process_result(controls):
    processed_result_exp, processed_result_history = {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history)
    if processed_result_exp:
        summarize_result(processed_result_exp)
    if processed_result_history:
        summarize_result(processed_result_history)
    return processed_result_exp, processed_result_history


def extract_result(control, model_tag, processed_result_exp, processed_result_history):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            # if 'test' not in base_result['logger']:
            # if 'train' in base_result['logger']:
            #     for k in base_result['logger']['train'].history:
            #         # metric_name = k.split('/')[1]
            #         metric_name = k
            #         if metric_name not in processed_result_history:
            #             processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
            #         # processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
            #         processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['train'].history[k]
            # else:
                # for k in base_result['logger']['test'].mean:
                #     metric_name = k.split('/')[1]
                #     if metric_name not in processed_result_exp:
                #         processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                #         processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                #     processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
                #     processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['train'].history[k]
                
            for k in base_result['logger']['test'].history:
                metric_name = k
                if metric_name not in processed_result_exp:
                    # processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}

                # processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['test'].history[k]
                if 'pq_indices' in metric_name:
                    a = base_result['logger']['test'].history[k]
                    b = 5
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]])
    return

# for standard error
def cal_se(std, sample_nums):
    return std / np.sqrt(sample_nums)

def summarize_result(processed_result):
    # print(f'processed_result: {processed_result}')
    if 'exp' in processed_result:
        pivot = 'exp'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
        processed_result['se'] = cal_se(np.std(processed_result[pivot], axis=0).item(), len(processed_result[pivot]))
        processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
        processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
        processed_result[pivot] = processed_result[pivot].tolist()
    elif 'history' in processed_result:
        pivot = 'history'
        filter_length = []
        for i in range(len(processed_result[pivot])):
            x = processed_result[pivot][i]
            if len(x) < 800:
                # print(f'len(x): {len(x)}')
                pass
            # print()
            # filter_length.append(x)
            # if len(x) > 10100:
            #     filter_length.append(x[800:])
            # else:
            filter_length.append(x)
            # elif len(processed_result[pivot][i]) == 801:
            #     filter_length.append(x[:800])
            # else:
            #     filter_length.append(x + [x[-1]] * (800 - len(x)))
        # print(processed_result[pivot])

        temp_length = []
        for item in filter_length:
            temp_length.append(len(item))

        processed_result[pivot] = filter_length
        a = copy.deepcopy(processed_result[pivot])
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        # TODO: for inference
        processed_result['se'] = cal_se(np.std(processed_result[pivot], axis=0), len(processed_result[pivot]))
        processed_result['max'] = np.max(processed_result[pivot], axis=1)
        processed_result['min'] = np.min(processed_result[pivot], axis=1)
        b = np.max(processed_result[pivot], axis=1)
        processed_result['mean_of_max'] = np.mean(np.max(processed_result[pivot], axis=1))
        processed_result['se_of_max'] = cal_se(np.std(np.max(processed_result[pivot], axis=1)), len(processed_result[pivot]))
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
        processed_result[pivot] = processed_result[pivot].tolist()
    else:
        for k, v in processed_result.items():
            # print(f'key {k}')
            # print(f'value length {len(v)}')
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result or 'history' in processed_result:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if exp_name not in extracted_processed_result:
            extracted_processed_result[exp_name] = defaultdict()
        
        extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[exp_name]['{}_se'.format(metric_name)] = processed_result['se']

        extracted_processed_result[exp_name]['{}_mean_of_max'.format(metric_name)] = processed_result['mean_of_max']
        extracted_processed_result[exp_name]['{}_se_of_max'.format(metric_name)] = processed_result['se_of_max']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def write_xlsx(path, df, startrow=0):
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return

def write_max_xlsx(path, df, startrow=0):
    writer = pd.ExcelWriter(path, engine='xlsxwriter')

    dataset_name = ['CIFAR10_', 'CIFAR100_', 'FEMNIST_']
    resource_ratios = ['0.3-0.7', '0.6-0.4', '0.9-0.1']
    diff_freqs = ['4-1', '5-1', '6-1']    

    server_ratio_client_ratio = []
    # for client resource constraint
    for ratio in resource_ratios:
        for freq in diff_freqs:
            server_ratio_client_ratio.append(f'1-0_{ratio}_{freq}')
    
    # for server resource constraint
    for ratio in resource_ratios:
        for freq in diff_freqs:
            server_ratio_client_ratio.append(f'{ratio}_1-0_{freq}')

    def write_one_dataset(cur_dataset, df, writer, server_ratio_client_ratio, startrow=0):
        # global result_path
        dynamicfl_name = []
        for df_name in df:
            if cur_dataset in df_name:
                # print(f'write {df_name}')
                if 'dynamicfl' in df_name:
                    dynamicfl_name.append(df_name)
                    continue
                
                sheet_name = cur_dataset            
                df[df_name] = pd.concat(df[df_name])
                df[df_name].to_excel(writer, sheet_name=sheet_name, startrow=startrow + 1)
                writer.sheets[sheet_name].write_string(startrow, 0, df_name)
                startrow = startrow + len(df[df_name].index) + 3
        
        # TODO: fix this
        if args['type'] == 'freq_ablation':
            return
        if args['type'] == 'communication_cost':
            return
        dynamicfl_name_sort = sorted(dynamicfl_name, key=lambda v: server_ratio_client_ratio.index(v[-15:]))

        while len(dynamicfl_name_sort) > 0:
            for df_name in df:
                if cur_dataset in df_name and 'dynamicfl' in df_name and dynamicfl_name_sort and dynamicfl_name_sort[0] in df_name:
                    sheet_name = cur_dataset            
                    df[df_name] = pd.concat(df[df_name])
                    df[df_name].to_excel(writer, sheet_name=sheet_name, startrow=startrow + 1)
                    writer.sheets[sheet_name].write_string(startrow, 0, df_name)
                    startrow = startrow + len(df[df_name].index) + 3
                    dynamicfl_name_sort.pop(0)

        return 
    
    for i in range(len(dataset_name)):
        cur_dataset = dataset_name[i]
        write_one_dataset(cur_dataset, df, writer, server_ratio_client_ratio)
        
    writer.save()
    return

def make_df_exp(extracted_processed_result_exp):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_exp:
        control = exp_name.split('_')
        if len(control) == 3:
            data_name, model_name, num_supervised = control
            index_name = ['1']
            df_name = '_'.join([data_name, model_name, num_supervised])
        elif len(control) == 10:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
            local_epoch, gm, sbn = control
            index_name = ['_'.join([local_epoch, gm])]
            df_name = '_'.join(
                [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, sbn])
        elif len(control) == 11:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
            local_epoch, gm, sbn, ft = control
            index_name = ['_'.join([local_epoch, gm])]
            df_name = '_'.join(
                [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, sbn,
                 ft])
        else:
            raise ValueError('Not valid control')
        df[df_name].append(pd.DataFrame(data=extracted_processed_result_exp[exp_name], index=index_name))
    write_xlsx('{}/result_exp.xlsx'.format(result_path), df)
    return df


def make_df_history(extracted_processed_result_history):
    df = defaultdict(list)
    df_of_max_non_iid_l_1 = defaultdict(list)
    df_of_max_non_iid_l_2 = defaultdict(list)
    df_of_max_non_iid_d_01 = defaultdict(list)
    df_of_max_non_iid_d_03 = defaultdict(list)


    for exp_name in extracted_processed_result_history:
        # print(f'exp_name: {exp_name}')
        control = exp_name.split('_')
        # print(f'len_control: {len(control)}')
        # dp
        if len(control) == 7:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch = control
            df_name = '_'.join(
                [data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch])
            for k in extracted_processed_result_history[exp_name]:
                index_name = ['_'.join([active_rate, data_split_mode, k])]
                # print(k)
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        # cnn & resnet18
        elif len(control) == 10:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            epoch, server_ratio, client_ratio, freq = control
            df_name = '_'.join(
                [data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
                    epoch, server_ratio, client_ratio, freq])
            
            max_mean_plus_se = []
            for k in extracted_processed_result_history[exp_name]:
                index_name = ['_'.join([active_rate, data_split_mode, k])]
                # print(f'k is: {k}')
                # print('\n')
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))

                if 'Accuracy' in k and 'mean_of_max' in k:
                    max_mean_plus_se.append(str(extracted_processed_result_history[exp_name][k].reshape(1, -1)[0][0]))
                # first is mean_of_max, second is std_of_max
                if 'Accuracy' in k and 'se_of_max' in k:
                    max_mean_plus_se.append('plus/minus')
                    # a = extracted_processed_result_history[exp_name][k].reshape(1, -1)
                    # b = extracted_processed_result_history[exp_name][k].reshape(1, -1)[0]
                    max_mean_plus_se.append(str(extracted_processed_result_history[exp_name][k].reshape(1, -1)[0][0]))
                    res = ' '.join(max_mean_plus_se)
                    if 'non-iid-l-1' in exp_name:
                        df_of_max_non_iid_l_1[df_name].append(
                        pd.DataFrame(data=[res], index=index_name))
                    elif 'non-iid-l-2' in exp_name:
                        df_of_max_non_iid_l_2[df_name].append(
                        pd.DataFrame(data=[res], index=index_name))
                    elif 'non-iid-d-0.1' in exp_name:
                        df_of_max_non_iid_d_01[df_name].append(
                        pd.DataFrame(data=[res], index=index_name))
                    elif 'non-iid-d-0.3' in exp_name:
                        df_of_max_non_iid_d_03[df_name].append(
                        pd.DataFrame(data=[res], index=index_name))
        elif len(control) == 11:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            epoch, relu_threshold, test_server_batch_size, delete_criteria, delete_threshold = control
            df_name = '_'.join(
                [data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
                    epoch, relu_threshold, test_server_batch_size, delete_criteria, delete_threshold])
            
            max_mean_plus_se = []
            for k in extracted_processed_result_history[exp_name]:
                index_name = ['_'.join([active_rate, data_split_mode, k])]
                # print(f'k is: {k}')
                # print('\n')
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        elif len(control) == 12:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            epoch, server_ratio, client_ratio, freq, _, _ = control
            df_name = '_'.join(
                [data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
                    epoch, server_ratio, client_ratio, freq, _, _])
            
            max_mean_plus_se = []
            for k in extracted_processed_result_history[exp_name]:
                index_name = ['_'.join([data_name, active_rate, data_split_mode, freq, server_ratio, client_ratio, k])]
                # print(f'k is: {k}')
                # print('\n')
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        
        elif len(control) == 13:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            epoch, relu_threshold, test_server_batch_size, prune_norm, delete_criteria, delete_threshold, batch_deletion = control
            df_name = '_'.join(
                [data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
                    epoch, relu_threshold, test_server_batch_size, prune_norm, delete_criteria, delete_threshold, batch_deletion])
            
            max_mean_plus_se = []
            for k in extracted_processed_result_history[exp_name]:
                index_name = ['_'.join([delete_criteria, delete_threshold, k])]
                # print(f'k is: {k}')
                # print('\n')
                # if 'Accuracy_mean' in k:
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        elif len(control) == 14:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            epoch, relu_threshold, test_server_batch_size, prune_norm, delete_method, delete_criteria, delete_threshold, batch_deletion = control
            df_name = '_'.join(
                [data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
                    epoch, relu_threshold, test_server_batch_size, prune_norm, delete_method, delete_criteria, delete_threshold, batch_deletion])
            
            max_mean_plus_se = []
            for k in extracted_processed_result_history[exp_name]:
                index_name = ['_'.join([delete_method, delete_criteria, delete_threshold, k])]
                # print(f'k is: {k}')
                # print('\n')
                # if 'Accuracy_mean' in k:
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
                # if 'Accuracy' in k and 'mean_of_max' in k:
                #     max_mean_plus_se.append(str(extracted_processed_result_history[exp_name][k].reshape(1, -1)[0][0]))
                # # first is mean_of_max, second is std_of_max
                # if 'Accuracy' in k and 'se_of_max' in k:
                #     max_mean_plus_se.append('plus/minus')
                #     # a = extracted_processed_result_history[exp_name][k].reshape(1, -1)
                #     # b = extracted_processed_result_history[exp_name][k].reshape(1, -1)[0]
                #     max_mean_plus_se.append(str(extracted_processed_result_history[exp_name][k].reshape(1, -1)[0][0]))
                #     res = ' '.join(max_mean_plus_se)
                #     if 'non-iid-l-1' in exp_name:
                #         df_of_max_non_iid_l_1[df_name].append(
                #         pd.DataFrame(data=[res], index=index_name))
                #     elif 'non-iid-l-2' in exp_name:
                #         df_of_max_non_iid_l_2[df_name].append(
                #         pd.DataFrame(data=[res], index=index_name))
                #     elif 'non-iid-d-0.1' in exp_name:
                #         df_of_max_non_iid_d_01[df_name].append(
                #         pd.DataFrame(data=[res], index=index_name))
                #     elif 'non-iid-d-0.3' in exp_name:
                #         df_of_max_non_iid_d_03[df_name].append(
                #         pd.DataFrame(data=[res], index=index_name))
        # if len(control) == 3:
        #     data_name, model_name, num_supervised = control
        #     index_name = ['1']
        #     for k in extracted_processed_result_history[exp_name]:
        #         df_name = '_'.join([data_name, model_name, num_supervised, k])
        #         df[df_name].append(
        #             pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        # elif len(control) == 10:
        #     data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
        #     local_epoch, gm, sbn = control
        #     index_name = ['_'.join([local_epoch, gm])]
        #     for k in extracted_processed_result_history[exp_name]:
        #         df_name = '_'.join(
        #             [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode,
        #              sbn, k])
        #         df[df_name].append(
        #             pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        # elif len(control) == 11:
        #     data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
        #     local_epoch, gm, sbn, ft = control
        #     index_name = ['_'.join([local_epoch, gm])]
        #     for k in extracted_processed_result_history[exp_name]:
        #         df_name = '_'.join(
        #             [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode,
        #              sbn, ft, k])
        #         df[df_name].append(
        #             pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        else:
            raise ValueError('Not valid control')
    write_xlsx('{}/result_history.xlsx'.format(result_path), df)
    # write_max_xlsx('{}/result_history_l_1_only_max.xlsx'.format(result_path), df_of_max_non_iid_l_1)
    # write_max_xlsx('{}/result_history_l_2_only_max.xlsx'.format(result_path), df_of_max_non_iid_l_2)
    # write_max_xlsx('{}/result_history_d_01_only_max.xlsx'.format(result_path), df_of_max_non_iid_d_01)
    # write_max_xlsx('{}/result_history_d_03_only_max.xlsx'.format(result_path), df_of_max_non_iid_d_03)
    return df


def transfer_freq_label(label):
    '''
    transfer frequency number label to frequency character label
    '''
    label_correspondence = {'6': 'a', '5': 'b', '4': 'c', '3': 'd', '2.5': 'e', '2': 'f', '1': 'g'}
    label_list = label.split('-')
    res = []
    for item in label_list:
        res.append(label_correspondence[item])
    return "-".join(res)

def transfer_data_heterogeneity(heterogeneity):

    heterogeneity_correspondence = {
        'non-iid-l-1': 'Non-IID-Class-1', 
        'non-iid-l-2': 'Non-IID-Classes-2',
        'non-iid-d-0.01': 'Non-IID-Dirichlet-0.01',
        'non-iid-d-0.1': 'Non-IID-Dirichlet-0.1',
        'non-iid-d-0.3': 'Non-IID-Dirichlet-0.3',
    }

    return heterogeneity_correspondence[heterogeneity]

def change_decimal_to_percentage(decimal):
    return '{:.2%}'.format(float(decimal))

def cut_decimal(decimal):
    decimal = float(decimal)
    return format(decimal, '.4f')

def make_vis(df_exp, df_history):
    # global result_path
    data_split_mode_dict = {'iid': 'IID', 'non-iid-l-2': 'Non-IID, $K=2$',
                            'non-iid-d-0.1': 'Non-IID, $\operatorname{Dir}(0.1)$',
                            'non-iid-d-0.3': 'Non-IID, $\operatorname{Dir}(0.3)$', 'fix-fsgd': 'DynamicSgd + FixMatch',
                            'fix-batch': 'FedAvg + FixMatch', 'fs': 'Fully Supervised', 'ps': 'Partially Supervised'}
    

    color = {'5_0.5': 'red', '1_0.5': 'orange', '5_0': 'dodgerblue', '5_0.9': 'blue', '5_0.5_nomixup': 'green',
             '5_0_nomixup': 'green', 'iid': 'red', 'non-iid-l-2': 'orange', 'non-iid-d-0.1': 'dodgerblue',
             'non-iid-d-0.3': 'green', 'fix-fsgd': 'red', 'fix-batch': 'blue',
             'fs': 'black', 'ps': 'orange',

             'ReLU': 'red',
             'Genetic': 'red',
             'DynaComm': 'orange',
             'Brute-force': 'green',

             'fedavg': 'purple',
             'fedprox': 'dodgerblue',
             'fedensemble': 'green',
             'scaffold': 'brown',
             'dynamicfl': 'red',
             'fedgen': 'pink',
             'feddyn': 'black',
             'fednova': 'orange',

             'dynamicfl_0.3-0.7': 'red',
             'dynamicfl_0.6-0.4': 'green',
             'dynamicfl_0.9-0.1': 'dodgerblue',

             'ReLU_0': 'red',
             'ReLU_0.01': 'green',
             'ReLU_0.02': 'dodgerblue',
             'ReLU_0.03': 'brown',
             'ReLU_0.04': 'orange',
             'ReLU_0.05': 'pink',
             'ReLU_0.06': 'black',
             'ReLU_0.07': 'purple',
             'ReLU_0.09': 'deeppink',
             'ReLU_0.1': 'teal',
             'ReLU_0.2': 'orangered',
             'ReLU_0.3': 'seagreen',
             'ReLU_0.4': 'indigo',
             'ReLU_0.5': 'blue',
             'ReLU_0.7': 'sienna',
             'ReLU_0.8': 'maroon',
             
            #  'sparsity_0': 'sienna',
            #  'sparsity_0.05': 'red',
            #  'sparsity_0.1': 'green',
            #  'sparsity_0.15': 'dodgerblue',
            #  'sparsity_0.2': 'brown',
            #  'sparsity_0.3': 'orange',
            #  'sparsity_0.4': 'black',
            #  'sparsity_0.5': 'purple',
             
            #  'PQ_0': 'sienna',
            #  'PQ_0.05': 'red',
            #  'PQ_0.1': 'green',
            #  'PQ_0.15': 'dodgerblue',
            #  'PQ_0.2': 'brown',
            #  'PQ_0.3': 'orange',
            #  'PQ_0.4': 'black',
            #  'PQ_0.5': 'purple',

             'sparsity_0': 'sienna',
             'sparsity_0.05': 'red',
             'sparsity_0.1': 'green',
             'sparsity_0.15': 'dodgerblue',
             'sparsity_0.2': 'brown',
             'sparsity_0.3': 'orange',
             'sparsity_0.4': 'black',
             'sparsity_0.5': 'purple',
             'sparsity_0.7': 'purple',
             
             'PQ_0': 'orange',
             'PQ_0.001': 'black',
             'PQ_0.01': 'brown',
             'PQ_0.03': 'crimson',
             'PQ_0.06': 'teal',
             'PQ_0.05': 'red',
             'PQ_0.07': 'red',
             'PQ_0.1': 'green',
             'PQ_0.15': 'dodgerblue',
             'PQ_0.2': 'brown',
             'PQ_0.3': 'orange',
             'PQ_0.4': 'black',
             'PQ_0.5': 'purple',
             'PQ_0.6': 'black',
             'PQ_0.7': 'purple',
             'PQ_0.8': 'sienna',
             'PQ_0.9': 'green',
             'PQ_1.0': 'red',
             'PQ_999': 'darkseagreen',

            # '0', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '999'
             'PQ': 'orange',
             'inter': 'purple',
             'union': 'red'   
             }
    linestyle = {'5_0.5': '-', '1_0.5': '--', '5_0': ':', '5_0.5_nomixup': '-.', '5_0_nomixup': '-.',
                 '5_0.9': (0, (1, 5)), 'iid': '-', 'non-iid-l-2': '--', 'non-iid-d-0.1': '-.', 'non-iid-d-0.3': ':',
                 'fix-fsgd': '--', 'fix-batch': ':', 'fs': '-', 'ps': '-.',


                'ReLU': (0, (3, 1, 1, 1, 1, 1)),
                'Genetic': (0, (1, 1)),
                'DynaComm': '--',
                'Brute-force': (0, (3, 1, 1, 1, 1, 1)),
                
                'fedavg': (0, (1, 1)),
                'fedprox': '--',
                'fedensemble': (0, (3, 1, 1, 1, 1, 1)),
                'scaffold': (5, (10, 3)),
                'dynamicfl': (0, (5, 1)),
                'fedgen': (0, (3, 1, 1, 1)),
                'feddyn': (0, (3, 1, 1, 1)),
                'fednova': (0, (3, 1, 1, 1)),

                'dynamicfl_0.3-0.7': (0, (5, 1)),
                'dynamicfl_0.6-0.4': (0, (3, 1, 1, 1)),
                'dynamicfl_0.9-0.1': (0, (3, 1, 1, 1)),

                'freq_6-0': (0, (5, 1)),
                'freq_6-5': (0, (1, 1)),
                'freq_6-4': '--',
                'freq_6-3': (0, (3, 1, 1, 1, 1, 1)),
                'freq_6-2.5': (5, (10, 3)),
                'freq_6-2': (0, (3, 1, 1, 1)),
                'freq_6-1': ':',
                
                'freq_5-0': (0, (5, 1)),
                'freq_5-4': (0, (1, 1)),
                'freq_5-3': '--',
                'freq_5-2.5': (0, (3, 1, 1, 1, 1, 1)),
                'freq_5-2': (5, (10, 3)),
                'freq_5-1': (0, (3, 1, 1, 1)),

                'freq_4-0': (0, (5, 1)),
                'freq_4-3': (0, (1, 1)),
                'freq_4-2.5': '--',
                'freq_4-2': (0, (3, 1, 1, 1, 1, 1)),
                'freq_4-1': (5, (10, 3)),

                'freq_3-0': (0, (5, 1)),
                'freq_3-2.5': (0, (1, 1)),
                'freq_3-2': '--',
                'freq_3-1': (0, (3, 1, 1, 1, 1, 1)),

                'sparsity_0': (0, (5, 5, 1)),
                'sparsity_0.05': (0, (5, 1)),
                'sparsity_0.1': (0, (1, 1)),
                'sparsity_0.15': '--',
                'sparsity_0.2': (0, (3, 1, 1, 1, 1, 1)),
                'sparsity_0.3': (5, (10, 3)),
                'sparsity_0.4': (0, (3, 1, 1, 1)),
                'sparsity_0.5': (0, (1, 1, 1)),

                'PQ_0': (0, (5, 5, 4)),
                'PQ_0.001': (6, (1, 1, 1, 1)),
                'PQ_0.01': (0, (2, 2, 2)),
                'PQ_0.03': (5, (5, 1)),
                'PQ_0.05': (10, (5, 1)),
                'PQ_0.06': (10, (5, 3)),
                'PQ_0.07': (10, (5, 3)),
                'PQ_0.1': (0, (1, 1)),
                'PQ_0.15': '--',
                'PQ_0.2': (0, (3, 1, 1, 1, 1, 1)),
                'PQ_0.3': (5, (10, 3)),
                'PQ_0.4': (0, (3, 1, 1, 1)),
                'PQ_0.5': (0, (1, 1, 10)),
                'PQ_0.6': (0, (1, 1, 5)),
                'PQ_0.7': (0, (1, 1, 1)),
                'PQ_0.8': '--',
                'PQ_0.9': (0, (5, 5, 1)),
                'PQ_1.0': (0, (3, 10, 1)),
                'PQ_999': (10, (3, 10, 1)),

                'ReLU_0': (0, (5, 5, 1)),
                'ReLU_0.01': (0, (5, 1)),
                'ReLU_0.02': (0, (1, 1)),
                'ReLU_0.03': '--',
                'ReLU_0.04': (0, (3, 1, 1, 1, 1, 1)),
                'ReLU_0.05': (0, (3, 1, 1, 1, 1, 1)),
                'ReLU_0.06': (5, (10, 3)),
                'ReLU_0.07': (0, (3, 1, 1, 1)),
                'ReLU_0.09': (0, (1, 1, 1)),
                'ReLU_0.1': (0, (5, 5, 1)),
                'ReLU_0.2': (0, (5, 1)),
                'ReLU_0.3': (0, (1, 1)),
                'ReLU_0.4': '--',
                'ReLU_0.5': (0, (3, 1, 1, 1, 1, 1)),
                'ReLU_0.7': (5, (10, 3)),
                'ReLU_0.8': (0, (3, 1, 1, 1)),

                'PQ': (5, (10, 3)),
                'inter': (0, (3, 1, 1, 1)),
                'union': (10, (2, 5))
                }
        
    resource_ratios = ['0.3-0.7', '0.6-0.4', '0.9-0.1']
    diff_freqs = ['6-1', '5-1', '4-1']
    color_for_dynamicfl = ['red', 'green', 'dodgerblue']
    linestyle_for_dynamicfl = [(0, (5, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1))]
    
    
    for ratio in resource_ratios:
        index = 0
        for freq in diff_freqs:
            new_key = f'dynamicfl_sameResourcediffFreq_{ratio}_{freq}'
            color[new_key] = color_for_dynamicfl[index]
            linestyle[new_key] = linestyle_for_dynamicfl[index]
            index += 1

    for freq in diff_freqs:
        index = 0
        for ratio in resource_ratios:
            new_key = f'dynamicfl_sameFreqdiffResource_{ratio}_{freq}'
            color[new_key] = color_for_dynamicfl[index]
            linestyle[new_key] = linestyle_for_dynamicfl[index]
            index += 1


    loc_dict = {'Accuracy': 'lower right', 'Loss': 'upper right'}
    fontsize = {'legend': 14, 'label': 14, 'ticks': 14, 'group_x_ticks': 8}
    metric_name = 'Accuracy'
    fig = {}
    reorder_fig = []

    dynamicfl_cost_mean_list = []
    dynamicfl_cost_se_list = []
    dynamicfl_cost_ratio_mean_list = []
    dynamicfl_cost_ratio_se_list = []

    dynamicfl_cost_name = []
    dynamicfl_cost_ratio_name = []
    for df_name in df_history:
        df_name_list = df_name.split('_')
        
        if len(df_name_list) == 13:
            # data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            #         epoch, relu_threshold, test_server_batch_size, delete_criteria, delete_threshold = df_name_list
            
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            epoch, relu_threshold, test_server_batch_size, prune_norm, delete_criteria, delete_threshold, batch_deletion = df_name_list
            
            def label_exists(plt, label):
                legend = plt.gca().legend_
                if legend:
                    existing_labels = [t.get_text() for t in legend.get_texts()]
                    return label in existing_labels
                return False

            def draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label='Activation Layers in Order', y_label='Accuracy'):
                # temp = range(len(x))            
                # plt.scatter(x, y, color=color[key_for_dict], linestyle=linestyle[key_for_dict], label=key_for_dict)
                if label_exists(plt, key_for_dict):
                    plt.scatter(x, y, color=color[key_for_dict], linestyle=linestyle[key_for_dict])
                else:
                    plt.scatter(x, y, color=color[key_for_dict], linestyle=linestyle[key_for_dict], label=key_for_dict)
                plt.plot(x, y, color=color[key_for_dict], linestyle=linestyle[key_for_dict])
                # plt.fill_between(x, (y - yerr), (y + yerr), color=color[algo_mode], alpha=.1)

                plt.errorbar(x, y, yerr=yerr, color=color[key_for_dict], linestyle=linestyle[key_for_dict])
                
                plt.xlabel(x_label, fontsize=fontsize['label'])
                plt.ylabel(y_label, fontsize=fontsize['label'])
                plt.xticks(fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
                plt.legend(loc=loc_dict['Accuracy'], fontsize=fontsize['legend'])
                return
            
            
            temp = df_history[df_name].iterrows()
            layer_dict = {
                'pq_indices': collections.defaultdict(list),
                'PQ_trans_delete_ratio_list': collections.defaultdict(list),
            }
            # for ((index, row), (index_se, row_se)) in zip(temp, temp):
            #     print('index: ', index)
            #     if 'pq_indices_mean' in index:
            #         a = 5
            #         b = row
            #         c = len(row)
            #         d = 'zz'

            # TODO: comment back for gradually deletion
            # for ((index, row), (index_se, row_se)) in zip(temp, temp):
            #     # print('index: ', index)

            #     # if 'mean' in index:
            #     #     continue

            #     if 'of_max' in index:
            #         continue
                
            #     # comment back for gradually deletion
            #     if 'pq_indices_mean' in index:
            #         if index not in layer_dict['pq_indices']:
            #             layer_dict['pq_indices'][index] = [None, None]
            #         temp_index = index
            #         layer_dict['pq_indices'][index][0] = row.tolist()

            #     if 'PQ_trans_delete_ratio_list_mean' in index:
            #         layer_dict['pq_indices'][temp_index][1] = row.tolist()
                
            #     # only for y: pq, x: eta

            # for key in layer_dict['pq_indices']:
            #     temp_key = key.split('/')[1]
            #     fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, test_server_batch_size, prune_norm, delete_criteria, delete_threshold, batch_deletion, temp_key, 'PQ_trans'])
            #     fig[fig_name] = plt.figure(fig_name)
            #     x = layer_dict['pq_indices'][key][1]
            #     y = layer_dict['pq_indices'][key][0]
            #     key_for_dict = batch_deletion
            #     draw_str_x_figure(plt, x, y, None, key_for_dict, 'delete_channel_ratio', 'PQ_index')

            
            for ((index, row), (index_se, row_se)) in zip(temp, temp):
                # print('index: ', index)

                # if 'mean' in index:
                #     continue

                if 'of_max' in index:
                    continue
                
                # only for y: pq, x: eta
                if 'pq_indices_mean' in index:
                    temp_key = index.split('/')[1]
                    fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, test_server_batch_size, prune_norm, delete_criteria, batch_deletion, temp_key, 'PQ_trans'])
                    fig[fig_name] = plt.figure(fig_name)
                    x = delete_threshold
                    y = row.tolist()[0]
                    key_for_dict = batch_deletion
                    draw_str_x_figure(plt, x, y, None, key_for_dict, 'eta', 'PQ_index')

                if 'delete_channel_ratio_mean' in index:
                    temp_key = index.split('/')[1]
                    fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, test_server_batch_size, prune_norm, delete_criteria, batch_deletion, temp_key, 'delete_channel_ratio'])
                    fig[fig_name] = plt.figure(fig_name)
                    x = delete_threshold
                    y = row.tolist()[0]
                    key_for_dict = batch_deletion
                    draw_str_x_figure(plt, x, y, None, key_for_dict, 'eta', 'delete_channel_ratio')

            # fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, test_server_batch_size, prune_norm, delete_criteria, batch_deletion, 'Accuracy'])
            # fig[fig_name] = plt.figure(fig_name)
            temp = df_history[df_name].iterrows()
            cur_accuracy = None
            for ((index, row), (index_se, row_se)) in zip(temp, temp):
                # print('index: ', index)
                if 'test/Accuracy_mean' not in index:
                    continue

                if 'of_max' in index:
                    continue
                
                key_for_dict = f'ReLU_{relu_threshold}'
                # label = 
                a = row.iloc[-1]

                x = float(delete_threshold)
                if float(delete_threshold) == 999:
                    x = -1
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
                cur_accuracy = row.iloc[0]
            #     draw_str_x_figure(plt, x, row.iloc[0], row_se.iloc[0], key_for_dict, 'eta_value')
            
            # if delete_criteria == 'PQ':
            #     fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, relu_threshold, test_server_batch_size, prune_norm, delete_criteria, batch_deletion, 'PQ_mean'])
            #     fig[fig_name] = plt.figure(fig_name)
            #     temp = df_history[df_name].iterrows()
            #     temp_list = []
            #     temp_se_list = []
            #     for ((index, row), (index_se, row_se)) in zip(temp, temp):
            #         # print('index: ', index)
            #         # a = row.iloc
            #         # a1 = len(a)
            #         # b = row[0]
            #         if 'test' not in index or 'PQ_index_mean' not in index:
            #             continue

            #         if 'of_max' in index:
            #             continue
                    
            #         print('PQ_index_mean', row[0])
            #         temp_list.append(np.mean(np.array(row[0])))
            #         temp_se_list.append(np.mean(np.array(row_se[0])))
            #     print('---\n')
            #     key_for_dict = f'{delete_criteria}_{delete_threshold}'
            #     draw_str_x_figure(plt, np.arange(len(temp_list)), temp_list, temp_se_list, key_for_dict, 'layers', 'PQ_mean')


            def draw_macs_acc_figure(plt, x, y, yerr, key_for_dict, annotation='default', x_label='Activation Layers in Order', y_label='Accuracy'):
                # temp = range(len(x))
                if label_exists(plt, key_for_dict):
                    plt.scatter(x, y, color=color[key_for_dict], linestyle=linestyle[key_for_dict])
                else:
                    plt.scatter(x, y, color=color[key_for_dict], linestyle=linestyle[key_for_dict], label=key_for_dict)
                # plt.plot(x, y, color=color[key_for_dict], linestyle=linestyle[key_for_dict])

                plt.annotate(annotation, (x, y))
                # plt.fill_between(x, (y - yerr), (y + yerr), color=color[algo_mode], alpha=.1)

                plt.errorbar(x, y, yerr=yerr, color=color[key_for_dict], linestyle=linestyle[key_for_dict])
                
                plt.xlabel(x_label, fontsize=fontsize['label'])
                plt.ylabel(y_label, fontsize=fontsize['label'])
                plt.xticks(fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
                plt.legend(loc=loc_dict['Accuracy'], fontsize=fontsize['legend'])
                return
            
            # if float(delete_threshold) != 999:
            fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, relu_threshold, test_server_batch_size, prune_norm, delete_criteria, 'MACs_ratio_mean'])
            fig[fig_name] = plt.figure(fig_name)
            temp = df_history[df_name].iterrows()
            temp_list = []
            temp_se_list = []

            for ((index, row), (index_se, row_se)) in zip(temp, temp):
                # print('index: ', index)
                if 'test' not in index or 'MACs_ratio_mean' not in index:
                    continue

                if 'of_max' in index:
                    continue
                
                # temp_list.append(np.mean(np.array(row[0])))
                # temp_se_list.append(np.mean(np.array(row_se[0])))
                MACs_ratio_mean = row[0]

            key_for_dict = batch_deletion
            # eta
            annotation = delete_threshold
            draw_macs_acc_figure(plt, MACs_ratio_mean, cur_accuracy, 0, key_for_dict, annotation, 'MACs_ratio_mean', 'Accuracy')

                # self.total_empty_channel_count_in_all_samples_ratio
                # self.empty_channel_count_in_all_samples_ratio
                # self.sparsity_ratio

                # TODO: fix
                # fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, relu_threshold, test_server_batch_size, delete_criteria, 'empty_whole_channel_count_in_all_samples_ratio'])
                # fig[fig_name] = plt.figure(fig_name)
                # temp = df_history[df_name].iterrows()
                # temp_list = []
                # for ((index, row), (index_se, row_se)) in zip(temp, temp):
                #     print('index: ', index)
                #     if 'test' not in index or 'empty_whole_channel_count_in_all_samples_ratio' not in index:
                #         continue

                #     if 'of_max' in index:
                #         continue
                    
                #     temp_list.append(np.mean(np.array(row[0])))
                # key_for_dict = f'{delete_criteria}_{delete_threshold}'
                # draw_str_x_figure(plt, np.arange(len(temp_list)), temp_list, 0, key_for_dict, 'layers', 'empty_whole_channel_count_in_all_samples_ratio')



                # TODO: comment back for number_of_pruned_channel
                # fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, relu_threshold, test_server_batch_size, delete_criteria, 'number_of_pruned_channel'])
                # fig[fig_name] = plt.figure(fig_name)
                # temp = df_history[df_name].iterrows()
                # temp_list = []
                # temp_se_list = []
                # for ((index, row), (index_se, row_se)) in zip(temp, temp):
                #     # print('index: ', index)
                #     if 'test' not in index or 'number_of_pruned_channel_mean' not in index:
                #         continue
                    
                #     # if 'total_empty_channel_count_in_all_samples_ratio' in index:
                #     #     continue

                #     if 'of_max' in index:
                #         continue
                    
                #     temp_list.append(np.mean(np.array(row[0])))
                #     temp_se_list.append(np.mean(np.array(row_se[0])))
                # key_for_dict = f'{delete_criteria}_{delete_threshold}'
                # draw_str_x_figure(plt, np.arange(len(temp_list)), temp_list, temp_se_list, key_for_dict, 'layers', 'number_of_pruned_channel')






            # fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, relu_threshold, test_server_batch_size, delete_criteria, 'sparsity_ratio'])
            # fig[fig_name] = plt.figure(fig_name)
            # temp = df_history[df_name].iterrows()
            # temp_list = []
            # for ((index, row), (index_se, row_se)) in zip(temp, temp):
            #     print('index: ', index)
            #     if 'test' not in index or 'sparsity_ratio' not in index:
            #         continue

            #     if 'of_max' in index:
            #         continue
                
            #     temp_list.append(np.mean(np.array(row[0])))
            # key_for_dict = f'{delete_criteria}_{delete_threshold}'
            # draw_str_x_figure(plt, np.arange(len(temp_list)), temp_list, 0, key_for_dict, 'layers', 'sparsity_ratio')
        if len(df_name_list) == 14:
            # data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            #         epoch, relu_threshold, test_server_batch_size, delete_criteria, delete_threshold = df_name_list
            
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            epoch, relu_threshold, test_server_batch_size, prune_norm, delete_method, delete_criteria, delete_threshold, batch_deletion = df_name_list
            
            def draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label='Activation Layers in Order', y_label='Accuracy'):
                # temp = range(len(x))
                plt.scatter(x, y, color=color[key_for_dict], linestyle=linestyle[key_for_dict], label=key_for_dict)
                plt.plot(x, y, color=color[key_for_dict], linestyle=linestyle[key_for_dict])
                # plt.fill_between(x, (y - yerr), (y + yerr), color=color[algo_mode], alpha=.1)

                plt.errorbar(x, y, yerr=yerr, color=color[key_for_dict], linestyle=linestyle[key_for_dict])
                
                plt.xlabel(x_label, fontsize=fontsize['label'])
                plt.ylabel(y_label, fontsize=fontsize['label'])
                plt.xticks(fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
                plt.legend(loc=loc_dict['Accuracy'], fontsize=fontsize['legend'])
                return

            fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, test_server_batch_size, prune_norm, delete_method, delete_criteria, batch_deletion, 'Accuracy'])
            fig[fig_name] = plt.figure(fig_name)
            temp = df_history[df_name].iterrows()
            for ((index, row), (index_se, row_se)) in zip(temp, temp):
                # print('index: ', index)
                if 'test/Accuracy' not in index:
                    continue

                if 'of_max' in index:
                    continue
                
                key_for_dict = f'ReLU_{relu_threshold}'
                # label = 
                a = row.iloc[-1]

                x = float(delete_threshold)
                if float(delete_threshold) == 999:
                    x = -1
                # plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
                draw_str_x_figure(plt, x, row.iloc[0], row_se.iloc[0], key_for_dict, 'eta_value')
            
            if delete_criteria == 'PQ':
                fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, relu_threshold, test_server_batch_size, prune_norm, delete_method, delete_criteria, batch_deletion, 'PQ_mean'])
                fig[fig_name] = plt.figure(fig_name)
                temp = df_history[df_name].iterrows()
                temp_list = []
                temp_se_list = []
                for ((index, row), (index_se, row_se)) in zip(temp, temp):
                    # print('index: ', index)
                    # a = row.iloc
                    # a1 = len(a)
                    # b = row[0]
                    if 'test' not in index or 'PQ_index_mean' not in index:
                        continue

                    if 'of_max' in index:
                        continue
                    
                    print('PQ_index_mean', row[0])
                    temp_list.append(np.mean(np.array(row[0])))
                    temp_se_list.append(np.mean(np.array(row_se[0])))
                print('---\n')
                key_for_dict = f'{delete_criteria}_{delete_threshold}'
                draw_str_x_figure(plt, np.arange(len(temp_list)), temp_list, temp_se_list, key_for_dict, 'layers', 'PQ_mean')


            if float(delete_threshold) != 999:
                fig_name = '_'.join([data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch, relu_threshold, test_server_batch_size, prune_norm, delete_method, delete_criteria, batch_deletion, 'delete_channel_ratio'])
                fig[fig_name] = plt.figure(fig_name)
                temp = df_history[df_name].iterrows()
                temp_list = []
                temp_se_list = []
                for ((index, row), (index_se, row_se)) in zip(temp, temp):
                    # print('index: ', index)
                    if 'test' not in index or 'delete_channel_ratio_mean' not in index:
                        continue

                    if 'of_max' in index:
                        continue
                    
                    temp_list.append(np.mean(np.array(row[0])))
                    temp_se_list.append(np.mean(np.array(row_se[0])))

                key_for_dict = f'{delete_criteria}_{delete_threshold}'
                draw_str_x_figure(plt, np.arange(len(temp_list)), temp_list, temp_se_list, key_for_dict, 'layers', 'delete_channel_ratio')    
        
    def write_xlsx(path, df, startrow=0):
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        for df_name in df:
            df[df_name] = pd.concat(df[df_name])
            df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
            writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
            startrow = startrow + len(df[df_name].index) + 3
        writer.save()
        return
    
    

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()


# dp_QL_mean_list = []
# dp_QL_se_list = []
# dp_QL_time_list = []
# dp_QL_time_se_list = []

# dp_KL_cc_03_mean_list = []
# dp_KL_cc_03_se_list = []
# dp_KL_cc_03_time_list = []
# dp_KL_cc_03_time_se_list = []

# dp_KL_cc_05_mean_list = []
# dp_KL_cc_05_se_list = []
# dp_KL_cc_05_time_list = []
# dp_KL_cc_05_time_se_list = []

# dp_KL_cc_07_mean_list = []
# dp_KL_cc_07_se_list = []
# dp_KL_cc_07_time_list = []
# dp_KL_cc_07_time_se_list = []

# dp_KL_cc_09_mean_list = []
# dp_KL_cc_09_se_list = []
# dp_KL_cc_09_time_list = []
# dp_KL_cc_09_time_se_list = []


# if 'time' in index:
#     if 'mean' in index:
#         if 'KL' in index and 'Brute-force' in index:
#             row = row + 1
#             Brute-force_KL_time_list.append(np.mean(np.log(row)))
#             Brute-force_KL_time_se_list.append(np.std(np.log(row)))
#         elif 'KL' in index and 'Genetic' in index:
#             row = row + 1
#             Genetic_KL_time_list.append(np.mean(np.log(row)))
#             Genetic_KL_time_se_list.append(np.std(np.log(row)))
#         elif 'KL' in index and 'dp' in index:
#             # avoid negative log
#             row = row + 1
#             # if f'1_communication_cost' in index:
#             #     dp_KL_cc_1_time_list.append(np.mean(np.log(row)))
#             #     dp_KL_cc_1_time_se_list.append(np.std(np.log(row)))
#             # elif f'3_communication_cost' in index:
#             #     dp_KL_cc_3_time_list.append(np.mean(np.log(row)))
#             #     dp_KL_cc_3_time_se_list.append(np.std(np.log(row)))
#             # elif f'5_communication_cost' in index:
#             #     dp_KL_cc_5_time_list.append(np.mean(np.log(row)))
#             #     dp_KL_cc_5_time_se_list.append(np.std(np.log(row)))
#             # else:
#             dp_KL_time_list.append(np.mean(np.log(row)))
#             dp_KL_time_se_list.append(np.std(np.log(row)))
#         # elif 'QL' in index and 'Brute-force' in index:
#         #     Brute-force_QL_time_list.append(np.mean(row))
#         #     Brute-force_QL_time_se_list.append(np.std(row))
#         # elif 'QL' in index and 'Genetic' in index:
#         #     Genetic_QL_time_list.append(np.mean(row))
#         #     Genetic_QL_time_se_list.append(np.std(row))
#         # elif 'QL' in index and 'dp' in index:
#         #     dp_QL_time_list.append(np.mean(row))
#         #     dp_QL_time_se_list.append(np.std(row))
# elif 'mean' in index:
# if 'KL' in index and 'Brute-force' in index:
#     Brute-force_KL_mean_list.append(np.mean(row))
#     Brute-force_KL_se_list.append(np.std(row))
# elif 'KL' in index and 'Genetic' in index:
#     Genetic_KL_mean_list.append(np.mean(row))
#     Genetic_KL_se_list.append(np.std(row))
# elif 'KL' in index and 'dp' in index:
#     # if f'0.3_ratio_communication_cost' in index:
#     #     dp_KL_cc_03_mean_list.append(np.mean(row))
#     #     dp_KL_cc_03_se_list.append(np.std(row))
#     # elif f'0.5_ratio_communication_cost' in index:
#     #     dp_KL_cc_05_mean_list.append(np.mean(row))
#     #     dp_KL_cc_05_se_list.append(np.std(row))
#     # elif f'0.7_ratio_communication_cost' in index:
#     #     dp_KL_cc_07_mean_list.append(np.mean(row))
#     #     dp_KL_cc_07_se_list.append(np.std(row))
#     # elif f'0.9_ratio_communication_cost' in index:
#     #     dp_KL_cc_09_mean_list.append(np.mean(row))
#     #     dp_KL_cc_09_se_list.append(np.std(row))
#     # else:
#     dp_KL_mean_list.append(np.mean(row))
#     dp_KL_se_list.append(np.std(row))
# # elif 'QL' in index and 'Brute-force' in index:
# #     Brute-force_QL_mean_list.append(np.mean(row))
# #     Brute-force_QL_se_list.append(np.std(row))
# # elif 'QL' in index and 'Genetic' in index:
# #     Genetic_QL_mean_list.append(np.mean(row))
# #     Genetic_QL_se_list.append(np.std(row))
# # elif 'QL' in index and 'dp' in index:
# #     dp_QL_mean_list.append(np.mean(row))
# #     dp_QL_se_list.append(np.std(row))


# # elif 'Quadratic' in index and 'Brute-force' in index:
# # mean_list.append(np.mean(row))
# # elif 'mean' in index:
# #     # index_list.append(index)
# #     if 'KL' in index and 'Genetic' in index:
# #         Genetic_KL_mean_list.append(np.mean(row))
# #     elif 'Quadratic_Loss' in index and 'Genetic' in index:
# #         Genetic_Quadratic_Loss_mean_list.append(np.mean(row))
# #     elif 'KL' in index and 'dp' in index:
# #         dp_KL_mean_list.append(np.mean(row))
# #     elif 'Quadratic_Loss' in index and 'dp' in index:
# #         dp_Quadratic_Loss_mean_list.append(np.mean(row))
# #     elif 'KL' in index and 'Brute-force' in index:
# #         Brute-force_KL_mean_list.append(np.mean(row))
# #     # elif 'Quadratic' in index and 'Brute-force' in index:
# #     # mean_list.append(np.mean(row))
# # elif 'std' in index:
# #     if 'KL' in index and 'Genetic' in index:
# #         Genetic_KL_se_list.append(np.mean(row))
# #     elif 'Quadratic_Loss' in index and 'Genetic' in index:
# #         Genetic_Quadratic_Loss_se_list.append(np.mean(row))
# #     elif 'KL' in index and 'dp' in index:
# #         dp_KL_se_list.append(np.mean(row))
# #     elif 'Quadratic_Loss' in index and 'dp' in index:
# #         dp_Quadratic_Loss_se_list.append(np.mean(row))
# #     elif 'KL' in index and 'Brute-force' in index:
# #         Brute-force_KL_se_list.append(np.mean(row))
# # else:
# #     raise ValueError('wrong index')