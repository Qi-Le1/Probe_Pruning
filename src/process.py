import os
import math
import itertools
import json
import copy
import numpy as np
import pandas as pd
from module import save, load, makedir_exist_ok
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
    if file == 'observe_llm':
        # ------- llama-2-7b
        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['3'], [f'magunstructglobal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]],
        #         ['full'], ['somemethods-3'], ['down-proj']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)
        
        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['3'], [f'magstructglobal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]],
        #         ['full'], ['somemethods-3'], ['down-proj']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'magstructlocal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]],
        #         ['full'], ['somemethods-3'], ['down-proj']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]],
        #         ['inter'], ['somemethods-3'], ['down-proj']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #         ['inter'], ['somemethods-3'], ['down-proj']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'w*pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #         ['inter'], ['somemethods-3'], ['down-proj']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [9999]],
        #         ['inter'], ['somemethods-3'], ['default', 'gate-proj+up-proj+down-proj', 'down-proj']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'w*pqstructlocal+h+2+{x}+-1+max' for x in [9999]],
        #         ['inter'], ['somemethods-3'], ['default', 'gate-proj+up-proj+down-proj', 'down-proj']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
        #             ['inter'], ['somemethods-3'], ['gate-proj', 'up-proj', 'down-proj', 'gate-proj+up-proj+down-proj']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'w*magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
        #             ['inter'], ['somemethods-3'], ['gate-proj', 'up-proj', 'down-proj', 'gate-proj+up-proj+down-proj']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
        #             ['inter'], ['somemethods-3'], ['gate-proj', 'up-proj', 'down-proj', 'gate-proj+up-proj+down-proj']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)
        
        control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
                    ['inter'], ['somemethods-3'], ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)
        # ----- opt 1.3b
        # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['3'], [f'magunstructglobal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #         ['full'], ['somemethods-3'], ['fc2']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)
        
        # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['3'], [f'magstructglobal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #         ['full'], ['somemethods-3'], ['fc2']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'magstructlocal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #         ['full'], ['somemethods-3'], ['fc2']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]],
        #         ['inter'], ['somemethods-3'], ['fc2']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #         ['inter'], ['somemethods-3'], ['fc2']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'w*pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #         ['inter'], ['somemethods-3'], ['fc2']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [9999]],
        #         ['inter'], ['somemethods-3'], ['default', 'fc2']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'w*pqstructlocal+h+2+{x}+-1+max' for x in [9999]],
        #         ['inter'], ['somemethods-3'], ['default', 'fc2']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
        #             ['inter'], ['somemethods-3'], ['fc1', 'fc2', 'fc1+fc2']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'w*magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
        #             ['inter'], ['somemethods-3'], ['fc1', 'fc2', 'fc1+fc2']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
        #             ['inter'], ['somemethods-3'], ['fc1', 'fc2', 'fc1+fc2']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
                    ['inter'], ['somemethods-3'], ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)
    elif file == 'observe_cv':
        # control_name = [[['CIFAR10', 'CIFAR100'], [ 'resnet18'], ['ic'], ['1'], [f'pqstructlocal:h:2:{x}:1:max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #                      ['inter'], ['somemethods-3'], ['default']]]
        # CIFAR10_controls_9 = make_controls( control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['ic'], ['1'], [f'w*pqstructlocal:h:2:{x}:1:max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #                     ['inter'], ['somemethods-3'], ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['ic'], ['1'], [f'magstructlocal:h:2:{x}:1:max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #                     ['inter'], ['somemethods-3'], ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['ic'], ['1'], [f'magstructlocal:w:2:{x}:1:max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #                     ['inter'], ['somemethods-3'], ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['ic'], ['1'], [f'magstructglobal:w:2:{x}:1:max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
        #                     ['inter'], ['somemethods-3'], ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['CIFAR10', 'CIFAR100'], [ 'resnet18'], ['ic'], ['1'], [f'pqstructlocal:h:2:{x}:1:max' for x in [9999]],
        #                      ['inter'], ['somemethods-3'], ['default']]]
        # CIFAR10_controls_9 = make_controls( control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['ic'], ['1'], [f'w*pqstructlocal:h:2:{x}:1:max' for x in [9999]],
        #                     ['inter'], ['somemethods-3'], ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)
        pass
    return controls


def main():
    global result_path, vis_path, num_experiments, exp

    print(f"type: {args['type']}")    
    result_path = './output/result/{}'.format(args['type'])
    vis_path = './output/vis/{}'.format(args['type'])
    files = [args['type']]

    if 'observe' in args['type']:
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
    extracted_processed_result_history = {}
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
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
        base_result_path_i = os.path.join(result_path, '{}'.format(model_tag))

        # print('result_path', result_path)
        # print('base_result_path_i', base_result_path_i)
        # for entry in os.listdir(result_path):
        #     print('entry', entry)
        # print('here', os.path.exists(base_result_path_i))
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
                # if 'norm_across_other_dims' in k:
                #     continue
                metric_name = k
                if metric_name not in processed_result_exp:
                    # processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}

                # processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['test'].history[k]
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
            # if len(x) > 500:
            #     continue
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
        # if len(filter_length) > 0:
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
            # if 'norm_across_other_dims' in k:
            #     print('kkkk', k)
            #     continue
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

def make_df_history(extracted_processed_result_history):
    df = defaultdict(list)

    for exp_name in extracted_processed_result_history:
        # print(f'exp_name: {exp_name}')
        control = exp_name.split('_')
        # print(f'len_control: {len(control)}')
        # dp
        if len(control) == 8:
            data_name, model_name, task_name, batch_size, prune_name, batch_integ, multibatch_integ, cust_tgt_modules = control
            df_name = '_'.join(
                [data_name, model_name, task_name, batch_size, prune_name, batch_integ, multibatch_integ, cust_tgt_modules])
            for k in extracted_processed_result_history[exp_name]:
                # print(f'k: {k}')
                index_name = ['_'.join([data_name, model_name, task_name, batch_size, prune_name, batch_integ, multibatch_integ, cust_tgt_modules, k])]
                # print(k)
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
                # df[df_name] = df[df_name].append(pd.DataFrame(...))
        else:
            raise ValueError('Not valid control')
    # TODO, dont write into xlsx right now
    # write_xlsx('{}/result_history.xlsx'.format(result_path), df)
    return df


def change_decimal_to_percentage(decimal):
    return '{:.2%}'.format(float(decimal))

def cut_decimal(decimal):
    decimal = float(decimal)
    return format(decimal, '.4f')

def label_exists(plt, label):
    legend = plt.gca().legend_
    if legend:
        existing_labels = [t.get_text() for t in legend.get_texts()]
        return label in existing_labels
    return False

def make_vis(df_exp, df_history):
    color = {
            # '0', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '999'
             'PQ': 'orange',
             'inter': 'purple',
             'union': 'red',
             'vanilla': 'green',
            'pruned': 'black'
             }
    linestyle = {'5_0.5': '-', '1_0.5': '--', '5_0': ':', '5_0.5_nomixup': '-.', '5_0_nomixup': '-.',
                 '5_0.9': (0, (1, 5)), 'iid': '-', 'non-iid-l-2': '--', 'non-iid-d-0.1': '-.', 'non-iid-d-0.3': ':',
                 'fix-fsgd': '--', 'fix-batch': ':', 'fs': '-', 'ps': '-.',
                'full': (5, (10, 3)),
                'inter': (0, (3, 1, 1, 1)),
                'union': (10, (2, 5)),
                'vanilla': '--',
                'pruned': '-.'
                }
        
    prune_hypers = [0, 0.001, 0.01, 0.03, 0.05, 0.06, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 9999]
    linestyle_patterns = {
        0: (0, (5, 5, 4)), 0.001: (6, (1, 1, 1, 1)), 0.01: (0, (2, 2, 2)), 0.03: (5, (5, 1)),
        0.05: (10, (5, 1)), 0.06: (10, (5, 3)), 0.07: (10, (5, 3)), 0.1: (0, (1, 1)),
        0.15: '--', 0.2: (0, (3, 1, 1, 1, 1, 1)), 0.3: (5, (10, 3)), 0.4: (0, (3, 1, 1, 1)),
        0.5: (0, (1, 1, 10)), 0.6: (0, (1, 1, 5)), 0.7: (0, (1, 1, 1)), 0.8: '--',
        0.9: (0, (5, 5, 1)), 1.0: (0, (3, 10, 1)),  9999: (10, (3, 10, 1))
    }
    color_patterns = {
        0: 'orange', 0.001: 'black', 0.01: 'brown', 0.03: 'crimson', 0.05: 'red', 
        0.06: 'teal', 0.07: 'red', 0.1: 'green', 0.15: 'dodgerblue', 0.2: 'brown', 
        0.3: 'orange', 0.4: 'black', 0.5: 'purple', 0.6: 'black', 0.7: 'purple', 
        0.8: 'sienna', 0.9: 'green', 1.0: 'red', 9999: 'darkseagreen'
    }
    prune_names = ['magstructglobal', 'magunstructglobal', 'pqstructlocal', 'w*pqstructlocal', 'magstructlocal', 'w*magstructlocal']
    total_layers = {
        'gpt2': 12,
        'opt-1.3b': 23,
        'llama-2-7b': 31,
        'llama-2': 31
    }
    for name in prune_names:
        for hyper in prune_hypers:
            linestyle[f"{name}_{hyper}"] = linestyle_patterns.get(hyper, (0, (1, 1)))
            color[f"{name}_{hyper}"] = color_patterns.get(hyper, 'orange')

    loc_dict = {'test/Rouge': 'lower right', 'test/ROUGE': 'lower right', 'test/GLUE': 'lower right', 'test/Accuracy': 'lower right', 'test/Perplexity': 'lower right', 'label': 'lower right'}
    fontsize = {'legend': 14, 'label': 14, 'ticks': 14, 'group_x_ticks': 8}
    metric_name_list = ['test/Rouge', 'test/ROUGE', 'test/GLUE', 'test/Accuracy', 'test/Perplexity']
    
    plot_layer_detail = False
    performance_metric_max = 70
    y_max_in_graph = 60
    fig = {}
    reorder_fig = []

    for df_name in df_history:
        df_name_list = df_name.split('_')
        if len(df_name_list) == 8:
            data_name, model_name, task_name, batch_size, prune_name, batch_integ, multibatch_integ, cust_tgt_modules = df_name_list
            
            prune_name_list = prune_name.split('+')
            prune_name = prune_name_list[0]
            prune_tgt = prune_name_list[1]
            if prune_tgt == 'w':
                prune_tgt = 'weight'
            elif prune_tgt == 'h':
                prune_tgt = 'hidden_repr'
            else:
                raise ValueError('Not valid prune target')
            prune_norm = prune_name_list[2]
            prune_hyper = prune_name_list[3]
            prune_dim = prune_name_list[4]
            prune_dim_select_mode = prune_name_list[5] if len(prune_name_list) > 5 else 'max'
            
            def is_valid_layer_for_detailed_info(index, model_name):
                if not plot_layer_detail:
                    return False

                if model_name in total_layers:
                    layer_number = int(index.split(".layers.")[1].split(".")[0])
                    if layer_number <= math.ceil(total_layers[model_name] * 0.05) or math.ceil(layer_number >= total_layers[model_name] * 0.95):
                        # print(f'layer_number: {layer_number}')
                        return True
                # print(f'False layer_number: {layer_number}')
                    return False
                else:
                    return True


            def draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label='Activation Layers in Order', y_label='Accuracy'):
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
                plt.legend(loc=loc_dict['label'], fontsize=fontsize['legend'])
                return
            
            def draw_histogram(plt, data, bins=500, density=False, x_label='Value', y_label='Frequency', title='Data Distribution'):
                plt.hist(data, bins=bins, density=density, color='blue', edgecolor='black')
                plt.xlabel(x_label, fontsize=12)
                plt.ylabel(y_label, fontsize=12)
                plt.title(title, fontsize=14)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                return

            def draw_bar(plt, bin_edges, data, x_label='Value', y_label='Prob', title='Data Distribution'):
                a = sum(data)
                b = np.diff(bin_edges)
                
                total = sum(data)
                density = [d / total for d in data] if total != 0 else data
                plt.bar(bin_edges[:-1], density, width=np.diff(bin_edges), align='edge')
                plt.xlabel(x_label, fontsize=12)
                plt.ylabel(y_label, fontsize=12)
                plt.title(title, fontsize=14)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                return



            def draw_macs_perform_figure(plt, x, y, yerr, key_for_dict, annotation='default', x_label='Activation Layers in Order', y_label='Accuracy'):
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
                plt.legend(loc=loc_dict['label'], fontsize=fontsize['legend'])
                return
            
            def process_distri(data, left_range=-1, right_range=1):
                bin_edges = [
                    -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, # -1000 to -100
                    -90, -80, -70, -60, -50, -40, -30, -20, -10,  
                    10, 20, 30, 40, 50, 60, 70, 80, 90, 100,  # -100 to 100 
                    200, 300, 400, 500, 600, 700, 800, 900, 1000  # 100 to 1000
                ]
                fine_bins = np.arange(-10, 10, 0.1).tolist()
                # for drawing 0
                bin_edges = bin_edges + fine_bins + [0.01]
                bin_edges = sorted(set(bin_edges))

                # Find the index for left_range and right_range
                left_index = next(i for i, x in enumerate(bin_edges) if x >= left_range)
                right_index = next(i for i, x in enumerate(bin_edges) if x >= right_range)
                return data[left_index:right_index+1], bin_edges[left_index:right_index+2]
            # if isinstance(df_history[df_name], list):
            #     # Handle the case where it's a list
            #     temp = df_history[df_name]
            # else:
                # It's not a list, so it can be treated as a DataFrame
            # print(type(df_history[df_name]))
            # print('here')
            # df_history[df_name] = pd.concat(df_history[df_name])
            # print('after concat')
            # b = a[0]
            # print(type(b))
            # temp = df_history[df_name].iterrows()  

            temp = copy.deepcopy(df_history[df_name]) 
            '''
                1. norm_across_other_dims distribution for each prune layer (y is number of shown, x is number, 1 for each layer)
                2. vanilla_hist distribution for each prune layer (y is number of shown, x is number, 1 for each layer)
                3. pruned_hist distribution for each prune layer (y is number of shown, x is number, 1 for each layer)
                2. pq_indices for each prune layer (y is pq_indices, x is all the layers, 1 for each prune_hyper)
                3. pruned_ratio for each prune layer (y is pruned_ratio, x is all the layers, 1 for each prune_hyper)
                4. pruned_FLOPs_ratio for each prune layer (y is pruned_FLOPs_ratio, x is all the layers, 1 for each prune_hyper)
                5. performance vs total_FLOPs_ratio (y is performance, x is total_FLOPs_ratio, all prune_hyper in 1)
            '''

            pq_indices_order = 0
            pruned_ratio_order = 0
            pruned_FLOPs_ratio_order = 0
            performance_vs_total_FLOPs_ratio = [None, None]
            performance_vs_sparsity = [None, None]
            for i in range(len(temp)):
                cur_item = temp[i]
                for index, row in cur_item.iterrows():
            # for ((index, row), (index_se, row_se)) in zip(temp, temp):
                    # print(f'index: {index}')
                    if 'of_max' in index:
                        continue
                    
                    if 'mean' not in index:
                        continue

                    index_list = index.split('/')
                    temp_key = index_list[-1]
                    if 'vanilla_hist_mean' in index:
                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # temp_key = index_list[-1]
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:', temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        data = row.tolist()
                        # draw_bar(plt, bin_edges[:-1], data)
                        temp_data, bin_edges = process_distri(data, -0.3, 1)
                        # draw_histogram(plt, data, bins=bin_edges, density=True)
                        draw_bar(plt, bin_edges, temp_data)
                        temp_data, bin_edges = process_distri(data, 0, 0.01)
                        # print('zzz', 0, temp_data[0])
                        y = temp_data[0]/(sum(data) + 1e-7)
                        plt.text(0, y, f'{y} for x=0', ha='center', va='bottom')

                    if 'pruned_hist_mean' in index:
                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # temp_key = index_list[-1]
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:',temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        data = row.tolist()
                        # draw_bar(plt, bin_edges[:-1], data)
                        temp_data, bin_edges = process_distri(data, -0.3, 1)
                        # draw_histogram(plt, data, bins=bin_edges, density=True)
                        draw_bar(plt, bin_edges, temp_data)
                        temp_data, bin_edges = process_distri(data, 0, 0.01)
                        y = temp_data[0]/(sum(data) + 1e-7)
                        plt.text(0, y, f'{y} for x=0', ha='center', va='bottom')

                    if 'norm_across_other_dims_mean' in index:
                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # temp_key = index_list[-1]
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:',temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        data = row.tolist()
                        # key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_histogram(plt, data)
                        zero_num = np.array(data)[np.array(data) == 0].shape[0]
                        plt.text(0, zero_num, f'{zero_num} for x=0', ha='center', va='bottom')
                    
                    if 'weight_norm_across_channel_dims' in index:
                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # temp_key = index_list[-1]
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:',temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        data = row.tolist()
                        # key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_histogram(plt, data)
                        zero_num = np.array(data)[np.array(data) == 0].shape[0]
                        plt.text(0, zero_num, f'{zero_num} for x=0', ha='center', va='bottom')
                    
                    # if 'pruned_channels_counter' in index:
                    #     fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:',temp_key])
                    #     fig[fig_name] = plt.figure(fig_name)
                    #     x = list(range(len(row.tolist())))
                    #     y = row.tolist()
                    #     # key_for_dict = f"{prune_name}_{prune_hyper}"
                    #     draw_str_x_figure(plt, x, y, None, key_for_dict, 'Channel index', 'Frequency')


                    if 'vanilla_duration_per_batch' in index or 'pruned_duration_per_batch' in index:
                        fig_name = '_'.join([data_name, model_name, task_name, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:', 'time_cost_per_batch'])
                        fig[fig_name] = plt.figure(fig_name)
                        x = batch_size
                        y = row.tolist()[0]
                        if 'vanilla_duration_per_batch' in index:
                            key_for_dict = "vanilla"
                        else:
                            key_for_dict = "pruned"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'Batch size', 'Seconds')


                    # only for y: pq, x: eta
                    if 'pq_indices_mean' in index:
                        # one prune_hyper for all layers (1 figure the whole model for each prune_hyper)
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:', 'all_layer_pq_indices_mean'])
                        fig[fig_name] = plt.figure(fig_name)
                        x = pq_indices_order
                        pq_indices_order += 1
                        y = row.tolist()[0]
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'Layer order', 'PQ_index')

                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # one layer for all prune_hyper (1 figure each layer for each prune_hyper)
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:', temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        x = prune_hyper
                        y = row.tolist()[0]
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'Eta', 'PQ_index')

                    if '_pq_lower_bound_mean' in index:
                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # one layer for all prune_hyper (1 figure each layer for each prune_hyper)
                        cur_temp_key = temp_key.replace('_pq_lower_bound_mean', '_pq_indices_varying_lengths_mean')
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:', cur_temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        x = int(row.tolist()[0])
                        print('lower_bound', x)
                        plt.text(x, 0.03, f'B', ha='center', va='bottom')

                    if "_pq_indices_varying_lengths_mean" in index:
                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # one layer for all prune_hyper (1 figure each layer for each prune_hyper)
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:', temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        x = list(range(len(row.tolist())))
                        y = np.minimum(np.array(row.tolist()), y_max_in_graph).tolist()
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'vector_length', 'PQ_index')

                        # one layer for all prune_hyper (1 figure each layer for each prune_hyper)
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG: log', temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        x = list(range(len(row.tolist())))
                        y = row.tolist()
                        y = np.log(y)
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'vector_length', 'PQ_index (log scale)')

                    if "reversed_pq_indices_varying_lengths" in index:
                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # one layer for all prune_hyper (1 figure each layer for each prune_hyper)
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:', temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        x = list(range(len(row.tolist())))
                        y = np.minimum(np.array(row.tolist()), y_max_in_graph).tolist()
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'vector_length(in reversed sorted)', 'PQ_index')
                    
                    if "_pq_indices_ratio" in index:
                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # one layer for all prune_hyper (1 figure each layer for each prune_hyper)
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:', temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        x = list(range(len(row.tolist())))
                        y = np.minimum(np.array(row.tolist()), y_max_in_graph).tolist()
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'small part(left) length', 'coefficient')

                    if f"_p_norm_ratio" in index:
                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # one layer for all prune_hyper (1 figure each layer for each prune_hyper)
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:', temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        x = list(range(len(row.tolist())))
                        y = np.minimum(np.array(row.tolist()), y_max_in_graph).tolist()
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'small part(left) length', 'coefficient')

                    if f"_q_norm_ratio" in index:
                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # one layer for all prune_hyper (1 figure each layer for each prune_hyper)
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:', temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        x = list(range(len(row.tolist())))
                        y = np.minimum(np.array(row.tolist()), y_max_in_graph).tolist()
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'small part(left) length', 'coefficient')

                    if 'pruned_ratio_mean' in index:
                        # one prune_hyper for all layers
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:','all_layer_pruned_ratio_mean'])
                        fig[fig_name] = plt.figure(fig_name)
                        x = pruned_ratio_order
                        pruned_ratio_order += 1
                        y = row.tolist()[0]
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'Layer order', 'pruned_ratio')

                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # one layer for all prune_hyper
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:',temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        x = prune_hyper
                        # pruned_ratio_order += 1
                        y = row.tolist()[0]
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'prune_hypers', 'pruned_ratio')

                    if 'pruned_FLOPs_ratio_mean' in index:
                        # one prune_hyper for all layers
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_hyper, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:','all_layer_pruned_FLOPs_ratio_mean'])
                        fig[fig_name] = plt.figure(fig_name)
                        x = pruned_FLOPs_ratio_order
                        pruned_FLOPs_ratio_order += 1
                        y = row.tolist()[0]
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'Layer order', 'pruned_FLOPs_ratio')

                        if not is_valid_layer_for_detailed_info(index, model_name):
                            continue
                        # one layer for all prune_hyper
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:',temp_key])
                        fig[fig_name] = plt.figure(fig_name)
                        x = prune_hyper
                        # pruned_FLOPs_ratio_order += 1
                        y = row.tolist()[0]
                        key_for_dict = f"{prune_name}_{prune_hyper}"
                        draw_str_x_figure(plt, x, y, None, key_for_dict, 'prune_hypers', 'pruned_FLOPs_ratio')
                    
                    


                    if any(metric_name in index for metric_name in metric_name_list) or 'total_FLOPs_ratio' in index:
                        cur_metric_name = next((metric for metric in metric_name_list if metric in index), None)
                        if any(metric_name in index for metric_name in metric_name_list):
                            print('metric')
                            if performance_vs_total_FLOPs_ratio[0] is None:
                                performance_vs_total_FLOPs_ratio[0] = min(performance_metric_max, row.tolist()[0])
                        elif 'total_FLOPs_ratio' in index:
                            if performance_vs_total_FLOPs_ratio[1] is None:
                                performance_vs_total_FLOPs_ratio[1] = row.tolist()[0]
                        
                        if performance_vs_total_FLOPs_ratio[0] is not None and performance_vs_total_FLOPs_ratio[1] is not None:
                            fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:performance_vs_total_FLOPs_ratio'])
                            fig[fig_name] = plt.figure(fig_name)
                            x = performance_vs_total_FLOPs_ratio[1]
                            y = performance_vs_total_FLOPs_ratio[0]
                            key_for_dict = f"{prune_name}_{prune_hyper}"
                            draw_macs_perform_figure(plt, x, y, 0, key_for_dict, prune_hyper, 'FLOPs_ratio', cur_metric_name)
                            performance_vs_total_FLOPs_ratio = [None, None]
                    
                    # data_name, model_name, task_name, batch_size, prune_name, batch_integ, multibatch_integ, cust_tgt_modules = df_name_list
                    # prune_name_list = prune_name.split('-')
                    # prune_name = prune_name_list[0]
                    # prune_tgt = prune_name_list[1]
                    # if prune_tgt == 'w':
                    #     prune_tgt = 'weight'
                    # elif prune_tgt == 'h':
                    #     prune_tgt = 'hidden_repr'
                    # else:
                    #     raise ValueError('Not valid prune target')
                    # prune_norm = prune_name_list[2]
                    # prune_hyper = prune_name_list[3]
                    # prune_dim = prune_name_list[4]
                    # prune_dim_select_mode = prune_name_list[5] if len(prune_name_list) > 5 else 'max'

                    # for unstructure pruning, performance v.s. sparsity
                    if any(metric_name in index for metric_name in metric_name_list) or 'sparsity' in index:
                        # print('here')
                        cur_metric_name = next((metric for metric in metric_name_list if metric in index), None)
                        if any(metric_name in index for metric_name in metric_name_list):
                            if performance_vs_sparsity[0] is None:
                                performance_vs_sparsity[0] = min(performance_metric_max, row.tolist()[0])
                        elif 'sparsity' in index:
                            if performance_vs_sparsity[1] is None:
                                performance_vs_sparsity[1] = row.tolist()[0]
                        print('performancevssparsity', performance_vs_sparsity)
                        if performance_vs_sparsity[0] is not None and performance_vs_sparsity[1] is not None:
                            # print('here1')
                            fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_name, prune_tgt, prune_norm, prune_dim, prune_dim_select_mode, batch_integ, multibatch_integ, cust_tgt_modules, 'FIG:performance_vs_sparsity'])
                            fig[fig_name] = plt.figure(fig_name)
                            x = performance_vs_sparsity[1]
                            y = performance_vs_sparsity[0]
                            key_for_dict = f"{prune_name}_{prune_hyper}"
                            draw_macs_perform_figure(plt, x, y, 0, key_for_dict, prune_hyper, 'sparsity', cur_metric_name)
                            performance_vs_sparsity = [None, None]


    def write_xlsx(path, df, startrow=0):
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        for df_name in df:
            df[df_name] = pd.concat(df[df_name])
            df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
            writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
            startrow = startrow + len(df[df_name].index) + 3
        writer.save()
        return

#     save_format = 'png'
# result_path = './output/result'
# vis_path = './output/vis/{}'.format(save_format)

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        # plt.figure(figsize=(10, 8))
        fig_name_list = fig_name.split('_')
        FIG_NAME = fig_name.split('FIG:')[-1]
        data_name = fig_name_list[0]
        model_name = fig_name_list[1]
        vis_path = os.path.join('output', 'vis', '{}'.format(save_format), data_name, model_name, FIG_NAME)
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()




# import os
# import itertools
# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# from module import save, load, makedir_exist_ok
# from collections import defaultdict

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# result_path = os.path.join('output', 'result')
# save_format = 'png'
# vis_path = os.path.join('output', 'vis', '{}'.format(save_format))
# num_experiments = 3
# exp = [str(x) for x in list(range(num_experiments))]
# dpi = 300
# matplotlib.rcParams['font.weight'] = 'bold'
# matplotlib.rcParams['axes.labelweight'] = 'bold'
# matplotlib.rcParams['axes.titleweight'] = 'bold'
# matplotlib.rcParams['axes.linewidth'] = 1.5
# matplotlib.rcParams['xtick.labelsize'] = 'large'
# matplotlib.rcParams['ytick.labelsize'] = 'large'
# matplotlib.rcParams["font.family"] = "Times New Roman"
# matplotlib.rcParams["font.serif"] = "Times New Roman"


# def make_controls(control_name):
#     control_names = []
#     for i in range(len(control_name)):
#         control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
#     controls = [exp] + [control_names]
#     controls = list(itertools.product(*controls))
#     return controls


# def make_all_controls(mode, task_name):
#     if task_name == 's2s':
#         data_names = ['fpb-sa', 'wikisql', 'samsum', 'e2enlg', 'webnlg-2017', 'dart']
#         model_names = ['bart-base']
#     elif task_name == 'clm':
#         data_names = ['dolly-15k']
#         model_names = ['llama-2']
#         # model_names = ['gpt2']
#     elif task_name == 'sc':
#         data_names = ['glue-cola', 'glue-mnli', 'glue-mrpc', 'glue-qnli', 'glue-qqp', 'glue-rte', 'glue-sst2',
#                       'glue-stsb']
#         model_names = ['roberta-base']
#     elif task_name == 'ic':
#         data_names = ['MNIST', 'CIFAR10']
#         model_names = ['linear', 'mlp', 'cnn']
#     else:
#         raise ValueError('Not valid task name')
#     if mode == 'full':
#         if task_name == 'ic':
#             batch_size = ['256']
#         else:
#             batch_size = ['32']
#         control_name = [[data_names, model_names, [task_name], ['full'], batch_size]]
#         controls = make_controls(control_name)
#     elif mode == 'peft':
#         if task_name == 'ic':
#             ft_name = ['lora']
#             batch_size = ['256']
#         else:
#             ft_name = ['lora', 'adalora', 'ia3', 'promptune', 'prefixtune', 'ptune']
#             if model_names[0] == 'llama-2':
#                 batch_size = ['8']
#             else:
#                 batch_size = ['32']
#         control_name = [[data_names, model_names, [task_name], ft_name, batch_size]]
#         controls = make_controls(control_name)
#     elif mode == 'cola':
#         ft_name = ['cola-lowrank-1', 'cola-linear-1', 'cola-mlp-1']
#         if task_name == 'ic':
#             batch_size = ['256']
#         else:
#             if model_names[0] == 'llama-2':
#                 batch_size = ['8']
#             else:
#                 batch_size = ['32']
#         control_name = [[data_names, model_names, [task_name], ft_name, batch_size]]
#         controls = make_controls(control_name)
#     elif mode == 'cola_step':
#         ft_name = ['cola-lowrank-1', 'cola-lowrank-2', 'cola-lowrank-4', 'cola-lowrank-8']
#         if task_name == 'ic':
#             batch_size = ['64']
#         else:
#             batch_size = ['8']
#         control_name = [[data_names, model_names, [task_name], ft_name, batch_size]]
#         controls = make_controls(control_name)
#     elif mode == 'cola_dist':
#         data_names = ['dolly-15k']
#         ft_name = ['cola-lowrank-1', 'cola-lowrank~linear-1', 'cola-lowrank~mlp-1']
#         if model_names[0] == 'llama-2':
#             batch_size = ['8']
#         else:
#             batch_size = ['32']
#         dist_mode = ['alone', 'col']
#         control_name = [[data_names, model_names, [task_name], ft_name, batch_size, dist_mode]]
#         controls = make_controls(control_name)
#     elif mode == 'cola_merge':
#         ft_name = ['cola-lowrank-1-1', 'cola-linear-1-1']
#         if task_name == 'ic':
#             batch_size = ['256']
#         else:
#             if model_names[0] == 'llama-2':
#                 batch_size = ['8']
#             else:
#                 batch_size = ['32']
#         control_name = [[data_names, model_names, [task_name], ft_name, batch_size]]
#         controls = make_controls(control_name)
#     elif mode == 'cola_dist_merge':
#         data_names = ['dolly-15k']
#         ft_name = ['cola-lowrank-1-1', 'cola-lowrank~linear-1-1']
#         if model_names[0] == 'llama-2':
#             batch_size = ['8']
#         else:
#             batch_size = ['32']
#         dist_mode = ['col']
#         control_name = [[data_names, model_names, [task_name], ft_name, batch_size, dist_mode]]
#         controls = make_controls(control_name)
#     else:
#         raise ValueError('Not valid mode')
#     return controls


# def main():
#     modes = ['full', 'peft', 'cola', 'cola_step', 'cola_dist', 'cola_merge', 'cola_dist_merge']
#     task_names = ['s2s', 'sc', 'clm', 'ic']
#     controls = []
#     for mode in modes:
#         for task_name in task_names:
#             controls += make_all_controls(mode, task_name)
#     processed_result = process_result(controls)
#     df_mean = make_df(processed_result, 'mean')
#     df_history = make_df(processed_result, 'history')
#     make_vis_method(df_history)
#     make_vis_step(df_history)
#     return


# def tree():
#     return defaultdict(tree)


# def process_result(controls):
#     result = tree()
#     for control in controls:
#         model_tag = '_'.join(control)
#         gather_result(list(control), model_tag, result)
#     summarize_result(None, result)
#     save(result, os.path.join(result_path, 'processed_result'))
#     processed_result = tree()
#     extract_result(processed_result, result, [])
#     return processed_result


# # def gather_result(control, model_tag, processed_result):
# #     if len(control) == 1:
# #         exp_idx = exp.index(control[0])
# #         base_result_path_i = os.path.join(result_path, '{}'.format(model_tag))
# #         if os.path.exists(base_result_path_i):
# #             base_result = load(base_result_path_i)
# #             for split in base_result['logger_state_dict']:
# #                 for metric_name in base_result['logger_state_dict'][split]['mean']:
# #                     processed_result[split][metric_name]['mean'][exp_idx] \
# #                         = base_result['logger_state_dict'][split]['mean'][metric_name]
# #                 for metric_name in base_result['logger_state_dict'][split]['history']:
# #                     processed_result[split][metric_name]['history'][exp_idx] \
# #                         = base_result['logger_state_dict'][split]['history'][metric_name]
# #         else:
# #             print('Missing {}'.format(base_result_path_i))
# #             pass
# #     else:
# #         gather_result([control[0]] + control[2:], model_tag, processed_result[control[1]])
# #     return


# def gather_result(control, model_tag, processed_result):
#     if len(control) == 1:
#         exp_idx = exp.index(control[0])
#         base_result_path_i = os.path.join(result_path, '{}'.format(model_tag))
#         if os.path.exists(base_result_path_i):
#             base_result = load(base_result_path_i)
#             for split in base_result['logger_state_dict']:
#                 for metric_name in base_result['logger_state_dict'][split]['mean']:
#                     processed_result[split][metric_name]['mean'][exp_idx] \
#                         = base_result['logger_state_dict'][split]['mean'][metric_name]
#                 for metric_name in base_result['logger_state_dict'][split]['history']:
#                     if 'info' in metric_name:
#                         continue
#                     x = base_result['logger_state_dict'][split]['history'][metric_name]
#                     if len(x) < 40 and len(x) > 10 and 'info' not in metric_name:
#                         # print('a', model_tag, len(x))
#                         num_miss = 40 - len(x)
#                         last_x = x[-1]
#                         x = x + [last_x + 1e-5 * np.random.randn() for _ in range(num_miss)]
#                     if len(x) < 10:
#                         print('b', model_tag, len(x))
#                         continue
#                     # processed_result[split][metric_name]['history'][exp_idx] \
#                     #     = base_result['logger_state_dict'][split]['history'][metric_name]
#                     processed_result[split][metric_name]['history'][exp_idx] = x
#         else:
#             print('Missing {}'.format(base_result_path_i))
#             pass
#     else:
#         gather_result([control[0]] + control[2:], model_tag, processed_result[control[1]])
#     return


# def summarize_result(key, value):
#     if key in ['mean', 'history']:
#         value['summary']['value'] = np.stack(list(value.values()), axis=0)
#         value['summary']['mean'] = np.mean(value['summary']['value'], axis=0)
#         value['summary']['std'] = np.std(value['summary']['value'], axis=0)
#         value['summary']['max'] = np.max(value['summary']['value'], axis=0)
#         value['summary']['min'] = np.min(value['summary']['value'], axis=0)
#         value['summary']['argmax'] = np.argmax(value['summary']['value'], axis=0)
#         value['summary']['argmin'] = np.argmin(value['summary']['value'], axis=0)
#         value['summary']['value'] = value['summary']['value'].tolist()
#     else:
#         for k, v in value.items():
#             summarize_result(k, v)
#         return
#     return


# def extract_result(extracted_processed_result, processed_result, control):
#     def extract(split, metric_name, mode):
#         output = False
#         if split == 'train':
#             if metric_name in ['test/Rouge', 'test/ROUGE', 'test/GLUE', 'test/Accuracy']:
#                 if mode == 'history':
#                     output = True
#         elif split == 'test':
#             if metric_name in ['test/Rouge', 'test/ROUGE', 'test/GLUE', 'test/Accuracy']:
#                 if mode == 'mean':
#                     output = True
#         elif split == 'test_each':
#             if metric_name in ['test/Rouge', 'test/ROUGE', 'test/GLUE', 'test/Accuracy']:
#                 if mode == 'mean':
#                     output = True
#         elif split == 'test_merge':
#             if metric_name in ['test/Rouge', 'test/ROUGE', 'test/GLUE', 'test/Accuracy']:
#                 if mode == 'mean':
#                     output = True
#         return output

#     if 'summary' in processed_result:
#         control_name, split, metric_name, mode = control
#         if not extract(split, metric_name, mode):
#             return
#         stats = ['mean', 'std']
#         for stat in stats:
#             exp_name = '_'.join([control_name, split, metric_name.split('/')[1], stat])
#             extracted_processed_result[mode][exp_name] = processed_result['summary'][stat]
#     else:
#         for k, v in processed_result.items():
#             extract_result(extracted_processed_result, v, control + [k])
#     return


# def make_df(processed_result, mode):
#     df = defaultdict(list)
#     for exp_name in processed_result[mode]:
#         exp_name_list = exp_name.split('_')
#         df_name = '_'.join([*exp_name_list])
#         index_name = [1]
#         df[df_name].append(pd.DataFrame(data=processed_result[mode][exp_name].reshape(1, -1), index=index_name))
#     startrow = 0
#     with pd.ExcelWriter(os.path.join(result_path, 'result_{}.xlsx'.format(mode)), engine='xlsxwriter') as writer:
#         for df_name in df:
#             df[df_name] = pd.concat(df[df_name])
#             df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1, header=False, index=False)
#             writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
#             startrow = startrow + len(df[df_name].index) + 3
#     return df


# def make_vis_method(df_history):
#     mode_name = ['full', 'lora', 'adalora', 'ia3', 'promptune', 'ptune', 'cola']
#     label_dict = {'full': 'FT', 'lora': 'LoRA', 'adalora': 'AdaLoRA', 'ia3': 'IA3', 'promptune': 'Promp Tuning',
#                   'prefixtune': 'Prefix Tuning', 'ptune': 'P-Tuning', 'cola-lowrank': 'ColA (Low Rank, unmerged)',
#                   'cola-linear': 'ColA (Linear, unmerged)', 'cola-mlp': 'ColA (MLP, unmerged)',
#                   'cola-lowrank-1': 'ColA (Low Rank, merged)', 'cola-linear-1': 'ColA (Linear, merged)'}
#     color_dict = {'full': 'black', 'lora': 'red', 'adalora': 'orange', 'ia3': 'green', 'promptune': 'blue',
#                   'prefixtune': 'dodgerblue', 'ptune': 'lightblue', 'cola-lowrank': 'gold',
#                   'cola-linear': 'silver', 'cola-mlp': 'purple', 'cola-lowrank-1': 'goldenrod',
#                   'cola-linear-1': 'gray'}
#     linestyle_dict = {'full': '-', 'lora': (0, (5, 5)), 'adalora': (0, (1, 1)), 'ia3': (0, (3, 5, 1, 5)),
#                       'promptune': (0, (5, 1)), 'prefixtune': (0, (1, 5)), 'ptune': (0, (5, 5, 1, 1)),
#                       'cola-lowrank': (0, (5, 1, 1, 1)), 'cola-linear': (0, (10, 5)), 'cola-mlp': (0, (10, 10)),
#                       'cola-lowrank-1': (0, (5, 5, 5, 1)), 'cola-linear-1': (0, (5, 10))}
#     marker_dict = {'full': 'D', 'lora': 's', 'adalora': 'p', 'ia3': 'd', 'promptune': 'd',
#                    'prefixtune': 'p', 'ptune': 's', 'cola-lowrank': 'o',
#                    'cola-linear': 'o', 'cola-mlp': 'o', 'cola-lowrank-1': 'o',
#                    'cola-linear-1': 'o', 'cola-mlp-1': 'o'}
#     loc_dict = {'ROUGE': 'lower right', 'GLUE': 'lower right', 'Accuracy': 'lower right'}
#     fontsize_dict = {'legend': 10, 'label': 16, 'ticks': 16}
#     figsize = (5, 4)
#     fig = {}
#     ax_dict_1 = {}
#     for df_name in df_history:
#         df_name_list = df_name.split('_')
#         model_name, mode, batch_size, metric_name, stat = df_name_list[1], df_name_list[3], df_name_list[4], \
#             df_name_list[-2], df_name_list[-1]
#         mask = len(df_name_list) - 3 == 5 and stat == 'mean'
#         if 'cola' in mode:
#             if model_name != 'llama-2' and batch_size not in ['32', '256']:
#                 mask = False
#             if model_name == 'llama-2' and len(df_name_list) - 3 == 5 and stat == 'mean':
#                 if mode.split('-')[2] == '1':
#                     mask = True
#                 else:
#                     mask = False
#         if mask:
#             df_name_std = '_'.join([*df_name_list[:-1], 'std'])
#             df_name_list[-2] = 'ROUGE' if df_name_list[-2] == 'Rouge' else df_name_list[-2]
#             fig_name = '_'.join([*df_name_list[:3], *df_name_list[4:-1]])
#             fig[fig_name] = plt.figure(fig_name, figsize=figsize)
#             if fig_name not in ax_dict_1:
#                 ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
#             ax_1 = ax_dict_1[fig_name]
#             y = df_history[df_name].iloc[0].to_numpy()
#             y_err = df_history[df_name_std].iloc[0].to_numpy()
#             x = np.arange(len(y))
#             xlabel = 'Epoch'
#             if 'cola' in mode:
#                 mode_list = mode.split('-')
#                 if len(mode_list) == 4 and mode_list[3] == '1':
#                     pivot = '-'.join([mode_list[0], mode_list[1], mode_list[3]])
#                 else:
#                     pivot = '-'.join([mode_list[0], mode_list[1]])
#             else:
#                 pivot = mode
#             metric_name = 'ROUGE' if metric_name == 'Rouge' else metric_name
#             ylabel = metric_name
#             ax_1.plot(x, y, label=label_dict[pivot], color=color_dict[pivot],
#                       linestyle=linestyle_dict[pivot])
#             ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict[pivot], alpha=.1)
#             ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
#             ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
#             ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
#             ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
#             ax_1.legend(loc=loc_dict[metric_name], fontsize=fontsize_dict['legend'])
#     for fig_name in fig:
#         fig_name_list = fig_name.split('_')
#         task_name = fig_name_list[2]
#         fig[fig_name] = plt.figure(fig_name)
#         ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
#         dir_name = 'method'
#         dir_path = os.path.join(vis_path, dir_name, task_name)
#         fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
#         makedir_exist_ok(dir_path)
#         plt.tight_layout()
#         plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
#         plt.close(fig_name)
#     return


# def make_vis_step(df_history):
#     mode_name = ['1', '2', '4', '8']
#     label_dict = {'1': '$I=1$', '2': '$I=2$', '4': '$I=4$', '8': '$I=8$'}
#     color_dict = {'1': 'black', '2': 'red', '4': 'orange', '8': 'gold'}
#     linestyle_dict = {'1': '-', '2': '--', '4': ':', '8': '-'}
#     marker_dict = {'1': 'D', '2': 's', '4': 'p', '8': 'o'}
#     loc_dict = {'ROUGE': 'lower right', 'GLUE': 'lower right', 'Accuracy': 'lower right'}
#     fontsize_dict = {'legend': 10, 'label': 16, 'ticks': 16}
#     figsize = (5, 4)
#     fig = {}
#     ax_dict_1 = {}
#     for df_name in df_history:
#         df_name_list = df_name.split('_')
#         model_name, method, batch_size, metric_name, stat = df_name_list[1], df_name_list[3], df_name_list[4], \
#             df_name_list[-2], df_name_list[-1]
#         mask = len(df_name_list) - 3 == 5 and stat == 'mean' and 'cola' in method
#         if ('cola-lowrank' not in method or len(method.split('-')) > 3
#                 or (model_name != 'llama-2' and batch_size not in ['8', '64'])):
#             mask = False
#         mode = method.split('-')[-1]
#         if mask:
#             df_name_std = '_'.join([*df_name_list[:-1], 'std'])
#             fig_name = '_'.join([*df_name_list[:3], *df_name_list[4:-1]])
#             fig[fig_name] = plt.figure(fig_name, figsize=figsize)
#             if fig_name not in ax_dict_1:
#                 ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
#             ax_1 = ax_dict_1[fig_name]
#             y = df_history[df_name].iloc[0].to_numpy()
#             # y_err = df_history[df_name_std].iloc[0].to_numpy()
#             y_err = 0
#             x = np.arange(len(y))
#             xlabel = 'Epoch'
#             pivot = mode
#             metric_name = 'ROUGE' if metric_name == 'Rouge' else metric_name
#             ylabel = metric_name
#             ax_1.plot(x, y, label=label_dict[pivot], color=color_dict[pivot],
#                       linestyle=linestyle_dict[pivot])
#             ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict[pivot], alpha=.1)
#             ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
#             ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
#             ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
#             ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
#             ax_1.legend(loc=loc_dict[metric_name], fontsize=fontsize_dict['legend'])
#     for fig_name in fig:
#         fig_name_list = fig_name.split('_')
#         fig[fig_name] = plt.figure(fig_name)
#         task_name = fig_name_list[2]
#         ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
#         dir_name = 'step'
#         dir_path = os.path.join(vis_path, dir_name, task_name)
#         fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
#         makedir_exist_ok(dir_path)
#         plt.tight_layout()
#         plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
#         plt.close(fig_name)
#     return


# if __name__ == '__main__':
#     main()


