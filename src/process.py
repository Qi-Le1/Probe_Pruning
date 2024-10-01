import os
import re
import math
import itertools
import json
import copy
import torch
import numpy as np
import pandas as pd
from module import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import random
import collections

plt.rcParams.update({
    'font.family': 'Nimbus Roman'
})
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='analyze_data')
parser.add_argument('--file', default='dp', type=str)
parser.add_argument('--detail', default='False', type=str)
parser.add_argument('--num_experiments', default=1, type=int)
args = vars(parser.parse_args())

save_format = 'png'
# result_path = './output/result'
result_path = f"./output/result/"
vis_path = './output/vis/{}'.format(save_format)

num_experiments = args['num_experiments']
exp = [str(x) for x in list(range(num_experiments))]


# for standard error
def cal_se(std, sample_nums):
    return std / np.sqrt(sample_nums)

# def change_decimal_to_percentage(decimal):
#     return '{:.2%}'.format(float(decimal))

# def cut_decimal(decimal):
#     decimal = float(decimal)
#     return format(decimal, '.1f')

def label_exists(plt, label):
    legend = plt.gca().legend_
    if legend:
        existing_labels = [t.get_text() for t in legend.get_texts()]
        return label in existing_labels
    return False

def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(file):
    controls = []
    # if file == 'similarchannel':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['ppwandasp'], ['probe'], ['sync'], ['None'], ['1-1-1-1-1-seqrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                         ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #     #                     ['flap'], ['probe-flap-default'], ['sync'], ['c4-2000'], ['1-1-1-1-1-seqrank'],
    #     #                 ['default']]]
    #     # CIFAR10_controls_9 = make_controls( control_name)
    #     # controls.extend(CIFAR10_controls_9)

    #     # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #     #                      ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #     #                     ['default']]]
    #     # CIFAR10_controls_9 = make_controls( control_name)
    #     # controls.extend(CIFAR10_controls_9)

    #     # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #     #                     ['flap'], ['probe-flapratio-bias'], ['sync'], ['c4-2000'], ['1-1-1-1-1-seqrank'],
    #     #                 ['default']]]
    #     # CIFAR10_controls_9 = make_controls( control_name)
    #     # controls.extend(CIFAR10_controls_9)
        


    #     # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
    #     #                 ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #     #             ['default']]]
    #     # CIFAR10_controls_9 = make_controls(control_name)
    #     # controls.extend(CIFAR10_controls_9)
        
    # if file == 'dense':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0'], 
    #                          ['None'], ['dense'], ['None'], ['None'], ['None'],        
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0'], 
    #                          ['None'], ['dense'], ['None'], ['None'], ['None'],        
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-13b'], ['csr'], ['20'], ['512'], ['0'], 
    #                          ['None'], ['dense'], ['None'], ['None'], ['None'],        
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['opt-13b'], ['csr'], ['20'], ['512'], ['0'], 
    #                          ['None'], ['dense'], ['None'], ['None'], ['None'],        
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'clm_task':

    #     # 'llama-2-7b', 'llama-2-13b', 'opt-13b'
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0'], 
    #                      ['None'], ['dense'], ['None'], ['None'], ['None'],        
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                      ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                      ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2',  '0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.4'], 
    #     #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-seqrank', '0.1-0.1-0.1-0.1-0.1-bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank', '0.05-0.05-0.05-0.05-0.05-bszrank'],
    #     #                 ['default']]]
    #     # CIFAR10_controls_9 = make_controls(control_name)
    #     # controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['ppwandasp'], ['probe'], ['sync'], ['None'], ['1-1-1-1-1-bszrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     # llama-3-8b
    #     control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0'], 
    #                      ['None'], ['dense'], ['None'], ['None'], ['None'],        
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls( control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     # llama-3-8b
    #     control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                      ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['gate-proj+up-proj+down-proj']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                      ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['gate-proj+up-proj+down-proj']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['gate-proj+up-proj+down-proj']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                         ['ppwandasp'], ['probe'], ['sync'], ['None'], ['1-1-1-1-1-bszrank'],
    #                     ['gate-proj+up-proj+down-proj']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'csr_task':

    #     # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0'], 
    #     #                      ['None'], ['dense'], ['None'], ['None'], ['None'],        
    #     #                     ['None']]]
    #     # CIFAR10_controls_9 = make_controls(control_name)
    #     # controls.extend(CIFAR10_controls_9)

    #     # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #     #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #     #                 ['default']]]
    #     # CIFAR10_controls_9 = make_controls(control_name)
    #     # controls.extend(CIFAR10_controls_9)

    #     # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #     #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #     #                 ['default']]]
    #     # CIFAR10_controls_9 = make_controls(control_name)
    #     # controls.extend(CIFAR10_controls_9)

    #     # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #     #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #     #                 ['default']]]
    #     # CIFAR10_controls_9 = make_controls(control_name)
    #     # controls.extend(CIFAR10_controls_9)

    #     # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #     #                  ['ppwandasp'], ['probe'], ['sync'], ['None'], ['1-1-1-1-1-bszrank'],
    #     #                 ['default']]]
    #     # CIFAR10_controls_9 = make_controls(control_name)
    #     # controls.extend(CIFAR10_controls_9)

    #     # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.4'], 
    #     #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-seqrank', '0.1-0.1-0.1-0.1-0.1-bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank', '0.05-0.05-0.05-0.05-0.05-bszrank'],
    #     #                 ['default']]]
    #     # CIFAR10_controls_9 = make_controls(control_name)
    #     # controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0'], 
    #                      ['None'], ['dense'], ['None'], ['None'], ['None'],        
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #                      ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['gate-proj+up-proj+down-proj']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #                      ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['gate-proj+up-proj+down-proj']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['gate-proj+up-proj+down-proj']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe'], ['sync'], ['None'], ['1-1-1-1-1-bszrank'],
    #                     ['gate-proj+up-proj+down-proj']]]
    #     CIFAR10_controls_9 = make_controls( control_name)
    #     controls.extend(CIFAR10_controls_9)

    # elif file == 'compare_metric':
    #     control_name = [[['wikitext-2v1'], ['llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.6'], 
    #                          ['wandasp', 'ppwandasp', 'flap'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
    #                         ['q-proj+k-proj+v-proj+o-proj', 'gate-proj+up-proj+down-proj', 'default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['opt-13b'], ['clm'], ['20'], ['1024'], ['0.6'], 
    #                         ['flap', 'wandasp', 'ppwandasp'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['q-proj+k-proj+v-proj+out-proj', 'fc1+fc2', 'default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'flap_calibration_compare':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000', 'wikivalid-2000'], ['None'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'inorderwiki':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['ppwandasp'], ['probe-default-inorderwiki'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'differentattentionmlp':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0-0', '0-0.2', '0-0.4', '0-0.6', '0.2-0', '0.2-0.2', '0.2-0.4', '0.2-0.6', '0.4-0', '0.4-0.2', '0.4-0.4', '0.4-0.6', '0.6-0', '0.6-0.2', '0.6-0.4', '0.6-0.6'], 
    #                          ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     # '0-0', '0.2-0.2','0.4-0.4','0.6-0.6'
    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0-0.2', '0.2-0', '0-0.4', '0.4-0', '0-0.6', '0.6-0', '0.2-0.4', '0.4-0.2', '0.2-0.6',  '0.6-0.2',  '0.4-0.6',   '0.6-0.4', ], 
    #                       ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                      ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'probeimproveperformance':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['wandasp', 'flap'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                          ['wandasp', 'flap'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'resinfo':
    #     # 'ppwandasp', 'wandasp'
    #     control_name = [[['wikitext-2v1'], ['llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.4', '0.6'], 
    #                      ['ppwandasp'], [ 'calib-resinfo0.9'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     # 'ppwandasp', 'wandasp'
    #     control_name = [[['arc-c'], ['llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.4', '0.6'], 
    #                      ['ppwandasp'], ['calib-resinfo0.9'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     # 'ppwandasp', 'wandasp'
    #     control_name = [[['wikitext-2v1'], ['llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.4', '0.6'], 
    #                      ['ppwandasp'], [ 'calib-resinfo1'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     # 'ppwandasp', 'wandasp'
    #     control_name = [[['arc-c'], ['llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.4', '0.6'], 
    #                      ['ppwandasp'], ['calib-resinfo1'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'llmpruner':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['None'], ['llmpruner-prune'], ['asyncinter'], ['None'], ['None'],
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #                         ['None'], ['llmpruner-prune'], ['asyncinter'], ['None'], ['None'],
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['None'], ['llmpruner-tune'], ['asyncinter'], ['None'], ['None'],
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #                         ['None'], ['llmpruner-tune'], ['asyncinter'], ['None'], ['None'],
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['None'], ['llmpruner-prune'], ['asyncinter'], ['None'], ['None'],
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-13b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #                         ['None'], ['llmpruner-prune'], ['asyncinter'], ['None'], ['None'],
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['None'], ['llmpruner-tune'], ['asyncinter'], ['None'], ['None'],
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-13b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #                         ['None'], ['llmpruner-tune'], ['asyncinter'], ['None'], ['None'],
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'loraprune':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['None'], ['loraprune-prune'], ['asyncinter'], ['None'], ['None'],
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #                         ['None'], ['loraprune-prune'], ['asyncinter'], ['None'], ['None'],
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['None'], ['loraprune-tune'], ['asyncinter'], ['None'], ['None'],
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #                         ['None'], ['loraprune-tune'], ['asyncinter'], ['None'], ['None'],
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['None'], ['loraprune-prune'], ['asyncinter'], ['None'], ['None'],
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-13b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #                         ['None'], ['loraprune-prune'], ['asyncinter'], ['None'], ['None'],
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['None'], ['loraprune-tune'], ['asyncinter'], ['None'], ['None'],
    #                         ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-13b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
    #                         ['None'], ['loraprune-tune'], ['asyncinter'], ['None'], ['None'],
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'inferencespeed':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0'], 
    #                      ['None'], ['dense'], ['None'], ['None'], ['None'],        
    #                     ['None']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                      ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)


    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2',  '0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'asyncintra':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
    #                          ['ppwandasp'], ['probe-default'], ['asyncintra'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-default'], ['asyncintra'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'flapsquare':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['flap'], ['flap-default-square'], ['asyncinter'], ['c4-2000'], ['None'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)        
    # elif file == 'calibvsnocalib':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
    #                          ['ppwandasp'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                          ['ppwandasp'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
    #                          ['ppwandasp'], ['calib-ema'], ['asyncinter'], ['c4-2000'], ['None'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                          ['ppwandasp'], ['calib-ema'], ['asyncinter'], ['c4-2000'], ['None'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
    #     #                  ['ppwandasp'], ['probe-respick'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank', '0.1-0.1-0.1-0.1-0.1-seqrank', '0.15-0.15-0.15-0.15-0.15-seqrank',\
    #     #                                                                                        '0.2-0.2-0.2-0.2-0.2-seqrank', '0.05-0.05-0.05-0.05-0.05-bszrank', '0.1-0.1-0.1-0.1-0.1-bszrank', '0.15-0.15-0.15-0.15-0.15-bszrank',\
    #     #                                                                                        '0.2-0.2-0.2-0.2-0.2-bszrank'],
    #     #                 ['default']]]
    #     # CIFAR10_controls_9 = make_controls(control_name)
    #     # controls.extend(CIFAR10_controls_9)
    # elif file == 'respickcompare':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-calib-ema'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-calib-ema'], ['asyncintra'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-default'], ['asyncintra'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'fixordynamic':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
    #                         ['ppwandasp'], ['probe-default-probefixratio0.5'], ['sync'], ['c4-2000'], 
    #                         ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)


    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                          ['ppwandasp'], ['probe-default-probefixratio0.5'], ['sync'], ['c4-2000'], 
    #                          ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
    #                         ['ppwandasp'], ['probe-default-probefixratio0.9'], ['sync'], ['c4-2000'], 
    #                         ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)


    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                          ['ppwandasp'], ['probe-default-probefixratio0.9'], ['sync'], ['c4-2000'], 
    #                          ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'channeldiff':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
    #                          ['ppwandasp'], ['probe-default-recorddiff'], ['sync'], ['c4-2000'], 
    #                 [ '0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
    # elif file == 'probevsprobenocalib':
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-respick'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.1-0.1-0.1-0.1-0.1-bszrank', '0.2-0.2-0.2-0.2-0.2-bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.1-0.1-0.1-0.1-0.1-bszrank', '0.2-0.2-0.2-0.2-0.2-bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['5'], ['1024'], ['0.2','0.4', '0.6'], 
    #                          ['ppwandasp'], ['probe-respick'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)
        
    #     control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['5'], ['1024'], ['0.2','0.4', '0.6'], 
    #                          ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                         ['default']]]
    #     CIFAR10_controls_9 = make_controls( control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-respick'], ['sync'], ['c4-2000'], 
    #                      ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.1-0.1-0.1-0.1-0.1-bszrank', '0.2-0.2-0.2-0.2-0.2-bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], 
    #                      ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.1-0.1-0.1-0.1-0.1-bszrank', '0.2-0.2-0.2-0.2-0.2-bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['5'], ['512'], ['0.2','0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-respick'], ['sync'], ['c4-2000'], 
    #                      ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    #     control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['5'], ['512'], ['0.2','0.4', '0.6'], 
    #                      ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], 
    #                      ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
    #                     ['default']]]
    #     CIFAR10_controls_9 = make_controls(control_name)
    #     controls.extend(CIFAR10_controls_9)

    # ---- for iclr
    if file == 'differentprobestudy':
        def probe_combination():
            return_list = []
            for seq in ['0.05', '0.1', '0.3',  '0.5',  '0.7', '0.8', '1.0']:
                for bsz in ['0.05', '0.25', '1']:
                # for bsz in ['1']:
                    # for rank in ['seqrank', 'bszrank']:
                        # Create the repeated sequence-bsz pattern
                    pattern = (f"{seq}+{bsz}-") * 5  # repeats 'seq+bsz-' five times
                    pattern = pattern[:-1]  # remove the last extra hyphen

                    # Append 'seqrank' and 'bszrank' placeholders or values
                    final_string = pattern + '-seqrank+bszrank'  # Adjust this as needed for actual rank values

                    # Append to the return list
                    return_list.append(final_string)
            return return_list

        control_name = [[['wikitext-2v1'], ['opt-13b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                        ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], probe_combination(),
                    ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                        ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], probe_combination(),
                    ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['arc-e'], ['opt-13b'], ['csr'], ['20'], ['512'], ['0.4'], 
                        ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], probe_combination(),
                    ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['arc-e'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.4'], 
                        ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], probe_combination(),
                    ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)


        control_name = [[['wikitext-2v1'], ['opt-13b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                        ['ppwandasp'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
                    ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                        ['ppwandasp'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
                    ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['arc-e'], ['opt-13b'], ['csr'], ['20'], ['512'], ['0.4'], 
                        ['ppwandasp'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
                    ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['arc-e'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.4'], 
                        ['ppwandasp'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
                    ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        
    elif file == 'recordcommonchannel':
        
        control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                             ['ppwandasp'], ['probe-recordcommonchannel'], ['sync'], ['None'], ['1-1-1-1-1-seqrank'],
                            ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
        #                      ['ppwandasp'], ['probe-recordcommonchannel-calib-ema'], ['sync'], ['c4-2000'], ['1-1-1-1-1-seqrank'],
        #                     ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                            ['ppwandasp'], ['probe-default-recordcommonchannel'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
                        ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                            ['ppwandasp'], ['calib-recordcommonchannel'], ['asyncinter'], ['c4-2000'], ['None'],
                        ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['arc-e'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.4'], 
                             ['ppwandasp'], ['probe-recordcommonchannel'], ['sync'], ['None'], ['1-1-1-1-1-seqrank'],
                            ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['arc-e'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.4'], 
                            ['ppwandasp'], ['probe-default-recordcommonchannel'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
                        ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['arc-e'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.4'], 
                            ['ppwandasp'], ['calib-recordcommonchannel'], ['asyncinter'], ['c4-2000'], ['None'],
                        ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)
        
        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
        #                      ['wandasp'], ['probe-recordcommonchannel'], ['sync'], ['None'], ['1-1-1-1-1-seqrank'],
        #                     ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
        #                      ['wandasp'], ['probe-recordcommonchannel-calib-ema'], ['sync'], ['c4-2000'], ['1-1-1-1-1-seqrank'],
        #                     ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
        #                     ['wandasp'], ['probe-default-recordcommonchannel'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
        #                 ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
        #                     ['wandasp'], ['calib-recordcommonchannel'], ['asyncinter'], ['c4-2000'], ['None'],
        #                 ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
        #                     ['flap'], ['probe-flap-flapratio-bias'], ['sync'], ['None'], ['1-1-1-1-1-seqrank'],
        #                 ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
        #                     ['flap'], ['probe-flap-flapratio-bias-respick-ema'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
        #                 ['default']]]
        # CIFAR10_controls_9 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_9)

        # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
        #                     ['flap'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
        #                 ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)
    elif file == 'compare_metric':
        control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                             ['wandasp', 'ppwandasp', 'flap'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
                            ['q-proj+k-proj+v-proj+o-proj', 'gate-proj+up-proj+down-proj', 'default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['wikitext-2v1'], ['opt-13b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                            ['flap', 'wandasp', 'ppwandasp'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
                        ['q-proj+k-proj+v-proj+out-proj', 'fc1+fc2', 'default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)
    elif file == 'csr_task':
        control_name = [[['boolq'], ['opt-13b'], ['csr'], ['20'], ['512'], ['0.2', '0.4'], 
                        ['ppwandasp'], ['probe'], ['sync'], ['c4-2000'], ['1-1-1-1-1-seqrank'],
                    ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)
    elif file == 'inferencespeed':
        control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                            ['flap'], ['flap-default-recordspeed'], ['asyncinter'], ['c4-2000'], ['None'],
                        ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                            ['wandasp'], ['calib-recordspeed'], ['asyncinter'], ['c4-2000'], ['None'],
                        ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                        ['ppwandasp'], ['probe-default-recordspeed'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
                    ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0'], 
                            ['None'], ['dense-recordspeed'], ['None'], ['None'], ['None'],        
                        ['None']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)
    elif file == 'ablationforfusion':
        def probe_combination():
            return_list = []
            for seq in ['0.05', '0.1', '0.3',  '0.5',  '0.7', '0.8', '1.0']:
                for bsz in ['0.05']:
                # for bsz in ['1']:
                    # for rank in ['seqrank', 'bszrank']:
                        # Create the repeated sequence-bsz pattern
                    pattern = (f"{seq}+{bsz}-") * 5  # repeats 'seq+bsz-' five times
                    pattern = pattern[:-1]  # remove the last extra hyphen

                    # Append 'seqrank' and 'bszrank' placeholders or values
                    final_string = pattern + '-seqrank+bszrank'  # Adjust this as needed for actual rank values

                    # Append to the return list
                    return_list.append(final_string)
            return return_list

        control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
                        ['ppwandasp'], ['probe-default-probefixratio0.1', 'probe-default-probefixratio0.5', 'probe-default-probefixratio0.9', 'probe-default'], ['sync'], ['c4-2000'], probe_combination(),
                    ['default']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)
    return controls


def main():
    global result_path, vis_path, num_experiments, exp
    file = args['file']
    vis_path = './output/vis/{}'.format(file)
    makedir_exist_ok(vis_path)

    controls = make_control_list(file)
    processed_result_history = {}
    process_result(controls, processed_result_history)
    
    # with open('{}/processed_result_exp.json'.format(result_path), 'w') as fp:
    #     json.dump(processed_result_exp, fp, indent=2)
    extracted_processed_result_history = {}
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    df_history = make_df_history(extracted_processed_result_history)
    make_vis(df_history)
    return

def check_missing_files(control, model_tag, processed_result_history):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}'.format(model_tag))
        if os.path.exists(base_result_path_i):
            pass
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_history:
            processed_result_history[control[1]] = {}
        check_missing_files([control[0]] + control[2:], model_tag, processed_result_history[control[1]])
    return

def extract_result(control, model_tag, processed_result_history):
    file = args['file']
    metric_name_list = ['test/Loss', 'test/Perplexity', 'test/CsrAccuracy', 'test/CsrAccuracyNorm']
    print('control', control)
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}'.format(model_tag))

        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)                
            for k in base_result['logger']['test'].history:
                if file == 'differentattentionmlp':
                    if any(metric_name in k for metric_name in metric_name_list) or 'fullinf_FLOPs_ratio_for_all_layers' in k:
                        if k not in processed_result_history:
                            processed_result_history[k] = {'history': [None for _ in range(num_experiments)]}
                        processed_result_history[k]['history'][exp_idx] = base_result['logger']['test'].history[k]
                elif file == 'resinfo':
                    if 'attn_sign_match_percentage' in k or 'attn_l2_magnitude_ratio' in k or 'attn_cosine_similarity' in k\
                        or 'mlp_sign_match_percentage' in k or 'mlp_l2_magnitude_ratio' in k or 'mlp_cosine_similarity' in k:
                        if k not in processed_result_history:
                            processed_result_history[k] = {'history': [None for _ in range(num_experiments)]}
                        processed_result_history[k]['history'][exp_idx] = base_result['logger']['test'].history[k]
                elif file == 'inferencespeed':
                    if 'inference_duration' in k:
                        if k not in processed_result_history:
                            processed_result_history[k] = {'history': [None for _ in range(num_experiments)]}
                        processed_result_history[k]['history'][exp_idx] = base_result['logger']['test'].history[k]
                elif file == 'flapsquare':
                    if any(metric_name in k for metric_name in metric_name_list) or 'average_pruning_ratio' in k:
                        if k not in processed_result_history:
                            processed_result_history[k] = {'history': [None for _ in range(num_experiments)]}
                        processed_result_history[k]['history'][exp_idx] = base_result['logger']['test'].history[k]
                elif file == 'channeldiff':
                    if 'diff_ratio' in k:
                        if k not in processed_result_history:
                            processed_result_history[k] = {'history': [None for _ in range(num_experiments)]}
                        processed_result_history[k]['history'][exp_idx] = base_result['logger']['test'].history[k]
                elif file == 'recordcommonchannel':
                    if 'cur_select_indices' in k:
                        if k not in processed_result_history:
                            processed_result_history[k] = {'history': [None for _ in range(num_experiments)]}
                        processed_result_history[k]['history'][exp_idx] = base_result['logger']['test'].history[k]
                else:
                    if any(metric_name in k for metric_name in metric_name_list):
                        print('kkkk', k, base_result['logger']['test'].history[k])
                        if k not in processed_result_history:
                            processed_result_history[k] = {'history': [None for _ in range(num_experiments)]}
                        processed_result_history[k]['history'][exp_idx] = base_result['logger']['test'].history[k]
                # if file == 'compare_metric' or file == 'clm_task' or file == 'csr_task':
                #     if any(metric_name in k for metric_name in metric_name_list):
                #         print('kkkk', k, base_result['logger']['test'].history[k])
                #         if k not in processed_result_history:
                #             processed_result_history[k] = {'history': [None for _ in range(num_experiments)]}
                #         processed_result_history[k]['history'][exp_idx] = base_result['logger']['test'].history[k]
                # else:
                #     if k not in processed_result_history:
                #         processed_result_history[k] = {'history': [None for _ in range(num_experiments)]}
                #     processed_result_history[k]['history'][exp_idx] = base_result['logger']['test'].history[k]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_history:
            processed_result_history[control[1]] = {}
        print('control[1]', control[1])
        extract_result([control[0]] + control[2:], model_tag, processed_result_history[control[1]])
    return

def summarize_result(processed_result, key):
    if 'history' in processed_result:
        pivot = 'history'
        results = []
        # for i in range(len(processed_result[pivot])):
        #     x = processed_result[pivot][i]
        #     results.append(x)
        print('processed_result[pivot]size', len(processed_result[pivot]))
        processed_result[pivot] = np.array(processed_result[pivot])
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        processed_result['std'] = np.std(processed_result[pivot], axis=0)
        processed_result['se'] = cal_se(processed_result['std'], len(processed_result[pivot]))
        # processed_result[pivot] = processed_result[pivot].tolist()
        # processed_result['max'] = np.max(processed_result[pivot], axis=1)
        # processed_result['min'] = np.min(processed_result[pivot], axis=1)
        # processed_result['mean_of_max'] = np.mean(np.max(processed_result[pivot], axis=1))
        # processed_result['std_of_max'] = np.std(np.max(processed_result[pivot], axis=1))
        # processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        # processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
    else:
        for k, v in processed_result.items():
            summarize_result(v, k)
        return
    return


def process_result(controls, processed_result_history):
    for control in controls:
        model_tag = '_'.join(control)
        check_missing_files(list(control), model_tag, processed_result_history)
    print(f'\n----- check missing {result_path} files done\n')
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_history)
    if processed_result_history:
        summarize_result(processed_result_history, None)
    return 

def trunc(values, decimal_places):
    values = np.round(values, decimal_places)
    values = np.trunc(values*10**decimal_places)/(10**decimal_places)
    return values

def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'history' in processed_result:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        print('metric_name', metric_name)
        if exp_name not in extracted_processed_result:
            extracted_processed_result[exp_name] = defaultdict()
        
        if 'average_pruning_ratio' in metric_name or 'diff_ratio' in metric_name:
            decimal_places = 4
        elif 'fullinf_FLOPs_ratio_for_all_layers' in metric_name:
            # delete probe flops
            processed_result['mean'] -= 0.015
            decimal_places = 2
        elif 'attn_sign_match_percentage' in metric_name or 'attn_l2_magnitude_ratio' in metric_name or 'attn_cosine_similarity' in metric_name\
                        or 'mlp_sign_match_percentage' in metric_name or 'mlp_l2_magnitude_ratio' in metric_name or 'mlp_cosine_similarity' in metric_name:
            decimal_places = 2
        else:
            decimal_places = 1
        if args['file'] == 'recordcommonchannel':
            extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = processed_result['history']
            # print("processed_result['history']", len(processed_result['history']), processed_result['history'])
            extracted_processed_result[exp_name]['{}_se'.format(metric_name)] = np.array(1)
        else:
            extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = trunc(processed_result['mean'], decimal_places)
            extracted_processed_result[exp_name]['{}_se'.format(metric_name)] = trunc(processed_result['se'], decimal_places)

        # extracted_processed_result[exp_name]['{}_se'.format(metric_name)] = np.round(processed_result['se'], 2)
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
    writer.close()
    return


def make_df_history(extracted_processed_result_history):
    df = defaultdict(list)
    df_for_xlsx = defaultdict(list)
    metric_name_list = ['test/Loss', 'test/Perplexity', 'test/CsrAccuracy', 'test/CsrAccuracyNorm']

    output_string = ''
    for exp_name in extracted_processed_result_history:
        control = exp_name.split('_')
        if len(control) == 12:
            data_name, model_name, task_name, batch_size, seq_len, prune_ratio, prune_metric, prune_method, mode,\
            calib_info, prune_info, cust_tgt_modules = control
            df_name = '_'.join(
                control)
            output_string += f'{df_name}\n'

            substring = ''
            for k in extracted_processed_result_history[exp_name]:
                # print('k', k)
                # index_name = ['_'.join(control + [k])]
                # df[df_name].append(
                #     pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
                
                # if 'fullinf_FLOPs_ratio_for_all_layers' in k or 'probe_FLOPs_ratio_for_all_layers' in k or \
                #     any(metric_name in k for metric_name in metric_name_list) or 'attn_sign_match_percentage' in k or 'attn_l2_magnitude_ratio' in k or 'attn_cosine_similarity' in k\
                #         or 'mlp_sign_match_percentage' in k or 'mlp_l2_magnitude_ratio' in k or 'mlp_cosine_similarity' in k or 'duration' in k or 'average_pruning_ratio' in k or 'diff_ratio' in k\
                #             or 'cur_select_indices' in k:
                #     print('inxlsxk', k, extracted_processed_result_history[exp_name][k].reshape(1, -1))
                #     if '_se' in k:
                #         value = extracted_processed_result_history[exp_name][k].reshape(1, -1)
                #         # value_type = type(value[0][0])
                #         # print('value_type', value_type, value[0][0])
                #         # # value[0][0] = f"({value[0][0]})"

                #         # value = pd.Series([f"({value[0][0]})"], dtype="string")

                #     else:
                #         value = extracted_processed_result_history[exp_name][k].reshape(1, -1)
                #     print('pdvalue', value)
                #     df_for_xlsx[df_name].append(
                #         pd.DataFrame(data=value, index=index_name))
                    
                #     if substring == '':
                #         substring += f'{index_name}: {value[0][0]}'
                #     else:
                #         substring += f'({value[0][0]})\n'
                #         output_string += substring
                #         substring = ''
                
                print('k', k)
                index_name = '_'.join(control + [k])

                cur_value = extracted_processed_result_history[exp_name][k]
                # if cur_value.ndim == 3:
                #     cur_value = cur_value[0]
                # else:
                #     cur_value = cur_value.reshape(1, -1)

                if df_name not in df:
                    df[df_name] = collections.defaultdict()
                
                df[df_name][index_name] = cur_value
                # df[df_name].append(
                #     pd.DataFrame(data=cur_value, index=index_name))
                
                if 'fullinf_FLOPs_ratio_for_all_layers' in k or 'probe_FLOPs_ratio_for_all_layers' in k or \
                    any(metric_name in k for metric_name in metric_name_list) or 'attn_sign_match_percentage' in k or 'attn_l2_magnitude_ratio' in k or 'attn_cosine_similarity' in k\
                        or 'mlp_sign_match_percentage' in k or 'mlp_l2_magnitude_ratio' in k or 'mlp_cosine_similarity' in k or 'inference_duration' in k or 'average_pruning_ratio' in k or 'diff_ratio' in k:
                    # print('inxlsxk', k, cur_value)
                    if '_se' in k:
                        value = cur_value
                        # value_type = type(value[0][0])
                        # print('value_type', value_type, value[0][0])
                        # # value[0][0] = f"({value[0][0]})"

                        # value = pd.Series([f"({value[0][0]})"], dtype="string")

                    else:
                        value = cur_value
                    # print('pdvalue', value)
                    # df_for_xlsx[df_name].append(
                    #     pd.DataFrame(data=value, index=index_name))
                    
                    if substring == '':
                        substring += f'{index_name}: {value[0]}'
                    else:
                        substring += f'({value[0]})\n'
                        output_string += substring
                        substring = ''
        else:
            raise ValueError('Not valid control')

    # write_xlsx(f"{result_path}/{args['file']}_result.xlsx", df_for_xlsx)
    if args['file'] != 'recordcommonchannel':
        file_path = f"{result_path}/{args['file']}_result.txt"
        # Open the file in write mode and write the code snippet
        with open(file_path, "w") as file:
            file.write(output_string)
    return df


def make_vis(df_history):
    file = args['file']
    color = {
             'probe': 'orange',
             'flap': 'purple',
             'wandasp': 'red',
             'dense': 'green',
             'llmpruner': 'blue',
             'llmprunerlora': 'brown',
             'Pruning Ratio 0.2': 'purple',
            'Pruning Ratio 0.4': 'red',
            'Pruning Ratio 0.6': 'green',
            'w/o PP': 'brown',
            'w/ PP': 'blue',
            'PP': 'orange',
             'FLAP': 'purple',
            'Probe Batch Size 1': 'orange',
            'Probe Batch Size 5': 'purple',
            'Probe Batch Size 10': 'red',
            'Probe Batch Size 20': 'green',

            'Fix-0.1': 'orange',
            'Fix-0.5': 'purple',
            'Fix-0.9': 'red',
            'Importance-Scaled Fusion': 'blue',
             }
    linestyle = {
                'probe': (5, (10, 3)),
                'flap': (0, (3, 1, 1, 1)),
                'PP': (5, (10, 3)),
                'FLAP': (0, (3, 1, 1, 1)),
                'wandasp': (10, (2, 5)),
                'dense': '--',
                'llmpruner': '-.',
                'llmprunerlora': (0, (1, 1)),
                'Pruning Ratio 0.2': (5, (10, 3)),
            'Pruning Ratio 0.4': (0, (3, 1, 1, 1)),
            'Pruning Ratio 0.6': (10, (2, 5)),

             'w/o PP': (5, (10, 3)),
            'w/ PP': (0, (3, 1, 1, 1)),

                'Probe Batch Size 1': (5, (10, 3)),
            'Probe Batch Size 5': (0, (3, 1, 1, 1)),
            'Probe Batch Size 10': (10, (2, 5)),
            'Probe Batch Size 20': '--',
                }
    marker = {
                'probe': 'D',
                'flap': 's',
                'PP': 'D',
                'FLAP': 's',
                'wandasp': 'H',
                'dense': '--',
                'llmpruner': 'x',
                'llmprunerlora': '+',
                'Pruning Ratio 0.2': 'o',
            'Pruning Ratio 0.4': 's',
            'Pruning Ratio 0.6': 'p',

            'w/o PP': 'o',
            'w/ PP': 's',

                'Probe Batch Size 1': 'o',
            'Probe Batch Size 5': 's',
            'Probe Batch Size 10': 'p',
            'Probe Batch Size 20': 'x',
                }
    # marker = {}
    # prune_ratios = [0, 0.001, 0.01, 0.03, 0.05, 0.06, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 9999]
    # linestyle_patterns = {
    #     0: (0, (5, 5, 4)), 0.001: (6, (1, 1, 1, 1)), 0.01: (0, (2, 2, 2)), 0.03: (5, (5, 1)),
    #     0.05: (10, (5, 1)), 0.06: (10, (5, 3)), 0.07: (10, (5, 3)), 0.1: (0, (1, 1)),
    #     0.15: '--', 0.2: (0, (3, 1, 1, 1, 1, 1)), 0.3: (5, (10, 3)), 0.4: (0, (3, 1, 1, 1)),
    #     0.5: (0, (1, 1, 10)), 0.6: (0, (1, 1, 5)), 0.7: (0, (1, 1, 1)), 0.8: '--',
    #     0.9: (0, (5, 5, 1)), 1.0: (0, (3, 10, 1)),  9999: (10, (3, 10, 1)),
    #     0.13: (0, (1, 1, 10)), 0.17: (0, (1, 1, 10)), 0.25: (0, (1, 1, 10)), 0.35: (0, (1, 1, 10)), 0.45: (0, (1, 1, 10)),
    #     'Our': (0, (1, 1, 10)), 'Mag':(6, (1, 1, 1, 1))
    # }
    # color_patterns = {
    #     0: 'orange', 0.001: 'black', 0.01: 'brown', 0.03: 'crimson', 0.05: 'red', 
    #     0.06: 'teal', 0.07: 'red', 0.1: 'green', 0.15: 'dodgerblue', 0.2: 'brown', 
    #     0.3: 'orange', 0.4: 'black', 0.5: 'purple', 0.6: 'black', 0.7: 'purple', 
    #     0.8: 'sienna', 0.9: 'green', 1.0: 'red', 9999: 'darkseagreen',
    #     0.13: 'orange', 0.17: 'green', 0.25: 'dodgerblue', 0.35: 'brown', 0.45: 'darkseagreen',
    #     'Our': 'orange', 'Mag': 'green'
    # }
    # prune_names = ['magstructglobal', 'magunstructglobal', 'pqstructlocal', 'w*pqstructlocal', 'magstructlocal', 'w*magstructlocal']
    # total_layers = {
    #     'gpt2': 12,
    #     'opt-1.3b': 23,
    #     'llama-2-7b': 31,
    #     'llama-2': 31
    # }
    # for name in prune_names:
    #     for hyper in prune_ratios:
    #         linestyle[f"{name}_{hyper}"] = linestyle_patterns.get(hyper, (0, (1, 1)))
    #         color[f"{name}_{hyper}"] = color_patterns.get(hyper, 'orange')

    backup_color_set = {'orange', 'green', 'red', 'purple', 'black', 'brown', 'blue', 'pink', 'teal','grey', 'cyan', 'magenta', 'yellow', 'indigo', 'silver', 'gold', 'seagreen', 'maroon', 'olive', 'lime', 'crimson', 'navy', 'olive', 'coral', 'steelblue', 'darkblue', 'darkviolet', 'slategray'}
    backup_linestyle_set = {(0, (3, 10, 1, 10)), (0, (3, 1, 2, 1)), '-.', (1, (5, 5)), (0, (1, 10)), (0, (5, 5, 4)), (6, (1, 1, 1, 1)), (0, (1, 1, 10)), (0, (2, 2, 2)), (5, (5, 1)), (10, (5, 1)), (10, (5, 3)),
                             (0, (1, 1)), '-.', '--', (2, (3, 5, 1, 5)), (1, (4, 10)), (3, (1, 1)), (3, (1, 2)), (3, (1, 3)), (3, (1, 4)), (3, (1, 5)), (3, (5, 10, 1)), (2, (5, 2, 1, 2)),(3, (5, 2, 1, 2)), (4, (5, 5, 1, 5)), (3, (1, 1, 1, 1)),(0, (1, 1, 15)), (0, (1, 2, 10)) }
    backup_marker_set = {'o', 'v', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '^', '<', '>', '1', '2', '3', '4', '+', '_', '|', 'x', '1', '2', '3', '4', '.', ',', (5, 0, 60), (6, 0, 49), (6, 0, 50)}

    # color['Proposed'] = 'orange'
    # color['State-of-the-art'] = 'green'
    # color['Full model'] = 'red'

    # # color['flap'] = 'orange'
    # # color['wandasp'] = 'green'
    # # color['pqnobias-0.5-0.5'] = 'red'
    # # color['pqnobiasglobal-0.5-0.5'] = 'purple'
    # # color['pqnobiasnormhead-0.5-0.5'] = 'brown'
    # linestyle['Proposed'] = (0, (1, 1, 10))
    # linestyle['State-of-the-art'] = (6, (1, 1, 1, 1))
    # linestyle['Full model'] = '--'
    # marker['Proposed'] = 'D'
    # marker['State-of-the-art'] = 's'
    # marker['Full model'] = '*'
    # linestyle['pqnobias-0.5-0.5'] = '-.'
    # linestyle['pqnobiasglobal-0.5-0.5'] = '-.'
    loc_dict = {'test/Perplexity': 'lower left', 'label': 'center right', 'test/CsrAccuracy': 'lower left', 'test/CsrAccuracyNorm': 'lower left'}
    fontsize = {'legend': 17, 'label': 13, 'ticks': 14, 'group_x_ticks': 8}
    # , 'test/CsrAccuracy', 'test/Loss',
    metric_name_list = [ 'test/Perplexity', 'test/CsrAccuracyNorm']

    fig = {}
    fig_data_across_multi_indices = collections.defaultdict(dict)


    def record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, **kwargs):
        if fig_name not in fig_data_across_multi_indices:
            fig_data_across_multi_indices[fig_name] = collections.defaultdict(dict)
        if key_for_dict not in fig_data_across_multi_indices[fig_name]:
            for key in kwargs:
                fig_data_across_multi_indices[fig_name][key_for_dict][key] = []
        
        for key in kwargs:
            fig_data_across_multi_indices[fig_name][key_for_dict][key].append(kwargs[key])
        return



    def draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label='Activation Layers in Order', y_label='Accuracy', label_loc=None, x_label_font_size=None, y_label_font_size=None, legend_size=None):
        fig_color = color.get(key_for_dict, random.choice(list(backup_color_set)))
        fig_linestyle = linestyle.get(key_for_dict, random.choice(list(backup_linestyle_set)))
        fig_marker = marker.get(key_for_dict, random.choice(list(backup_marker_set)))
        if key_for_dict not in color:
            color[key_for_dict] = fig_color
            linestyle[key_for_dict] = fig_linestyle
            marker[key_for_dict] = fig_marker

            backup_color_set.remove(fig_color)
            backup_linestyle_set.remove(fig_linestyle)
            backup_marker_set.remove(fig_marker)
        if label_exists(plt, key_for_dict):
            print('label exists', key_for_dict)
            return
            # plt.scatter(x, y, color=fig_color, linestyle=fig_linestyle)
        # else:
        #     plt.scatter(x, y, color=fig_color, linestyle=fig_linestyle, label=key_for_dict)
            # plt.scatter(x, y, color=fig_color, linestyle=fig_linestyle)
        plt.plot(x, y, color=fig_color, linestyle=fig_linestyle, marker=fig_marker, label=key_for_dict)
        # plt.fill_between(x, (y - yerr), (y + yerr), color=color[algo_mode], alpha=.1)
        plt.errorbar(x, y, yerr=yerr, color=fig_color, linestyle=fig_linestyle)
        if x_label_font_size:
            x_fontsize = x_label_font_size
        else:
            x_fontsize = fontsize['label']
        
        if y_label_font_size:
            y_fontsize = y_label_font_size
        else:
            y_fontsize = fontsize['label']

        plt.xlabel(x_label, fontsize=x_fontsize)
        plt.ylabel(y_label, fontsize=y_fontsize)
        plt.xticks(fontsize=fontsize['ticks'])
        plt.yticks(fontsize=fontsize['ticks'])
        if label_loc:
            loc = label_loc
        else:
            loc = loc_dict['label']
        
        if legend_size:
            legend_size = legend_size
        else:
            legend_size = fontsize['legend']
        plt.legend(loc=loc, fontsize=legend_size)
        # plt.legend(loc=loc_dict['label'], fontsize=fontsize['legend'], bbox_to_anchor=(1, 0.5))
        return
    
    def draw_scatter_x_figure(plt, x, y, yerr, key_for_dict, x_label='Activation Layers in Order', y_label='Accuracy'):
        fig_color = color.get(key_for_dict, random.choice(list(backup_color_set)))
        fig_linestyle = linestyle.get(key_for_dict, random.choice(list(backup_linestyle_set)))
        fig_marker = marker.get(key_for_dict, random.choice(list(backup_marker_set)))
        if key_for_dict not in color:
            color[key_for_dict] = fig_color
            linestyle[key_for_dict] = fig_linestyle
            marker[key_for_dict] = fig_marker

            backup_color_set.remove(fig_color)
            backup_linestyle_set.remove(fig_linestyle)
            backup_marker_set.remove(fig_marker)
        if label_exists(plt, key_for_dict):
            print('label exists', key_for_dict)
            return
            # plt.scatter(x, y, color=fig_color, linestyle=fig_linestyle)
        # else:
        plt.scatter(x, y, color=fig_color, linestyle=fig_linestyle, marker=fig_marker, label=key_for_dict)
            # plt.scatter(x, y, color=fig_color, linestyle=fig_linestyle)
        # plt.plot(x, y, color=fig_color, linestyle=fig_linestyle, marker=fig_marker, label=key_for_dict)
        # plt.fill_between(x, (y - yerr), (y + yerr), color=color[algo_mode], alpha=.1)
        plt.errorbar(x, y, yerr=yerr, color=fig_color, linestyle=fig_linestyle)
        plt.xlabel(x_label, fontsize=fontsize['label'])
        plt.ylabel(y_label, fontsize=fontsize['label'])
        plt.xticks(fontsize=fontsize['ticks'])
        plt.yticks(fontsize=fontsize['ticks'])
        plt.legend(loc=loc_dict['label'], fontsize=fontsize['legend'], bbox_to_anchor=(1, 0.5))
        return
    
    
    

    def draw_bar(plt, bin_edges, data, x_label='Value', y_label='Prob', title='Data Distribution'):
        a = sum(data)
        b = np.diff(bin_edges)
        
        total = sum(data)
        density = [d / total for d in data] if total != 0 else data
        plt.bar(bin_edges[:-1], density, width=np.diff(bin_edges), align='edge')
        plt.xlabel(x_label, fontsize=fontsize['label'])
        plt.ylabel(y_label, fontsize=fontsize['label'])
        # plt.title(title, fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        return



    def draw_macs_perform_figure(plt, x, y, yerr, key_for_dict, x_label='Activation Layers in Order', y_label='Accuracy', y_lim=None):
        fig_color = color.get(key_for_dict, random.choice(list(backup_color_set)))
        fig_linestyle = linestyle.get(key_for_dict, random.choice(list(backup_linestyle_set)))
        if key_for_dict not in color:
            color[key_for_dict] = fig_color
            linestyle[key_for_dict] = fig_linestyle

            backup_color_set.remove(fig_color)
            backup_linestyle_set.remove(fig_linestyle)
        # temp = range(len(x))
        if label_exists(plt, key_for_dict):
            plt.scatter(x, y, color=fig_color, linestyle=fig_linestyle)
        else:
            plt.scatter(x, y, color=fig_color, linestyle=fig_linestyle, label=key_for_dict)
        plt.plot(x, y, color=color, linestyle=linestyle)
        # TODO: comment out
        # plt.annotate(annotation, (x, y))
        # plt.fill_between(x, (y - yerr), (y + yerr), color=color[algo_mode], alpha=.1)

        plt.errorbar(x, y, yerr=yerr, color=fig_color, linestyle=fig_linestyle)
        if y_lim:
            plt.ylim(0, y_lim+5)
        plt.xlabel(x_label, fontsize=fontsize['label'])
        plt.ylabel(y_label, fontsize=fontsize['label'])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc=loc_dict['label'], fontsize=fontsize['legend'], bbox_to_anchor=(1, 0.5))
        return
    
        
    def find_layer_number(index):

        layer_number = index.split('/')[1]

        # Find all numbers in the string
        layer_number = re.findall(r'\d+', layer_number)

        # Join the number strings and convert to an integer
        layer_number = int(''.join(layer_number)) if layer_number else None
        return layer_number
    
    def get_probe_type_ratio(prune_info, batch_size):
        batch_size = int(batch_size)
        probe_seq_ratio = None
        probe_batch_size = None
        prune_info_list = prune_info.split('-')
        probe_type_list = prune_info_list[-1].split('+')
        prune_info_sub_list = prune_info_list[0].split('+')
        for i in range(len(probe_type_list)):
            probe_type = probe_type_list[i]
            ratio = float(prune_info_sub_list[i])
            
            if 'seq' in probe_type:
                probe_seq_ratio = ratio
            elif 'bsz' in probe_type:
                probe_batch_size = int(ratio * batch_size)

        return probe_seq_ratio, probe_batch_size


    select_channels_indices = {}
    for df_name in df_history.keys():
        loss_performance_vs_total_FLOPs_ratio = [None, None, None]
        ppl_performance_vs_total_FLOPs_ratio = [None, None, None]
        performance_vs_prunedflops = [None, None, None]
        dense_time_vs_total_FLOPs_ratio = [None, None, None]
        prune_time_vs_total_FLOPs_ratio = [None, None, None]

        df_name_list = df_name.split('_')
        # print('df_name_list', df_name_list, len(df_name_list))
        if len(df_name_list) == 12:
            data_name, model_name, task_name, batch_size, seq_len, prune_ratio, prune_metric, prune_method, mode,\
            calib_info, prune_info, cust_tgt_modules = df_name_list

            # prune_name_list = prune_name.split('+')
            # prune_name = prune_name_list[0]
            # if len(prune_name_list) == 1:
            #     nsamples = 'fulldata'
            # else:
            #     nsamples = prune_name_list[1]
            #     prune_name = prune_name + '+' + nsamples

            # temp = copy.deepcopy(df_history[df_name]) 
            # '''
            #     1. norm_across_other_dims distribution for each prune layer (y is number of shown, x is number, 1 for each layer)
            #     2. dense_hist distribution for each prune layer (y is number of shown, x is number, 1 for each layer)
            #     3. pruned_hist distribution for each prune layer (y is number of shown, x is number, 1 for each layer)
            #     2. pq_indices for each prune layer (y is pq_indices, x is all the layers, 1 for each prune_ratio)
            #     3. pruned_ratio for each prune layer (y is pruned_ratio, x is all the layers, 1 for each prune_ratio)
            #     4. pruned_FLOPs_ratio for each prune layer (y is pruned_FLOPs_ratio, x is all the layers, 1 for each prune_ratio)
            #     5. performance vs total_FLOPs_ratio (y is performance, x is total_FLOPs_ratio, all prune_ratio in 1)
            # '''

            # pq_indices_order = 0
            # pruned_ratio_order = 0
            # pruned_FLOPs_ratio_order = 0
            # norm_across_other_dims = None
            # temp_norm_across_other_dims_key = None


            # fullinf_vs_optimal_select_mean_intersection_ratio_order = 0
            # probe_vs_optimal_select_mean_intersection_ratio_order = 0
            # probe_vs_fullinf_select_mean_intersection_ratio_order = 0
            # fullinf_vs_optimal_prune_mean_intersection_ratio_order = 0
            # probe_vs_optimal_prune_mean_intersection_ratio_order = 0
            # probe_vs_fullinf_prune_mean_intersection_ratio_order = 0
            
            # for i in range(0, len(temp), 2):
            #     cur_item = temp[i]
            #     cur_se_item = temp[i+1]
            #     for ((index, row), (index_std, row_se)) in zip(cur_item.iterrows(), cur_se_item.iterrows()):

            # temp = copy.deepcopy(df_history[df_name]) 
            '''
                1. norm_across_other_dims distribution for each prune layer (y is number of shown, x is number, 1 for each layer)
                2. dense_hist distribution for each prune layer (y is number of shown, x is number, 1 for each layer)
                3. pruned_hist distribution for each prune layer (y is number of shown, x is number, 1 for each layer)
                2. pq_indices for each prune layer (y is pq_indices, x is all the layers, 1 for each prune_ratio)
                3. pruned_ratio for each prune layer (y is pruned_ratio, x is all the layers, 1 for each prune_ratio)
                4. pruned_FLOPs_ratio for each prune layer (y is pruned_FLOPs_ratio, x is all the layers, 1 for each prune_ratio)
                5. performance vs total_FLOPs_ratio (y is performance, x is total_FLOPs_ratio, all prune_ratio in 1)
            '''

            pq_indices_order = 0
            pruned_ratio_order = 0
            pruned_FLOPs_ratio_order = 0
            norm_across_other_dims = None
            temp_norm_across_other_dims_key = None


            fullinf_vs_optimal_select_mean_intersection_ratio_order = 0
            probe_vs_optimal_select_mean_intersection_ratio_order = 0
            probe_vs_fullinf_select_mean_intersection_ratio_order = 0
            fullinf_vs_optimal_prune_mean_intersection_ratio_order = 0
            probe_vs_optimal_prune_mean_intersection_ratio_order = 0
            probe_vs_fullinf_prune_mean_intersection_ratio_order = 0
            
            cur_dict = df_history[df_name]
            keys = list(cur_dict.keys())

            # Iterate over the dictionary by index, stepping by 2 to get pairs
            for i in range(0, len(keys) - 1, 2):  # Ensures we don't go out of range
                index, index_std = keys[i], keys[i + 1]
                row, row_se = cur_dict[index], cur_dict[index_std]
                # print('row', row)
            # for i in range(0, len(temp), 2):
            #     cur_item = temp[i]
            #     cur_se_item = temp[i+1]
            #     for ((index, row), (index_std, row_se)) in zip(cur_item.iterrows(), cur_se_item.iterrows()):
                print('index before', index)
                if 'mean' not in index:
                    continue
                
                index_list = index.split('/')
                temp_key = index_list[-1]

                # layer_number = find_layer_number(index)
                # if layer_number > 5:
                #     continue
                # print('index', index)
                # if 'gridsearch' in prune_method and ('Loss' in index or 'fullinf_FLOPs_ratio_for_all_layers' in index):
                #     if any(metric_name in index for metric_name in metric_name_list):
                #         flops_metric_name = next((metric for metric in metric_name_list if metric in index), None)
                #         flops_metric_name = flops_metric_name.split('/')[1]
                #         if loss_performance_vs_total_FLOPs_ratio[0] is None:
                #             loss_performance_vs_total_FLOPs_ratio[0] = row.tolist()[0]
                #             loss_performance_vs_total_FLOPs_ratio[1] = row_se.tolist()[0]
                #     elif 'fullinf_FLOPs_ratio_for_all_layers' in index :
                #         if loss_performance_vs_total_FLOPs_ratio[2] is None:
                #             loss_performance_vs_total_FLOPs_ratio[2] = row.tolist()[0]

                    
                #     if loss_performance_vs_total_FLOPs_ratio[0] is not None and loss_performance_vs_total_FLOPs_ratio[2] is not None:
                #         print('performancevssparsity', loss_performance_vs_total_FLOPs_ratio, flops_metric_name, prune_ratio)
                #         fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, prune_metric, prune_method, mode,\
                #         calib_info, prune_info, cust_tgt_modules, 'FIG:Loss_fullinf_FLOPs_ratio_for_all_layers'])
                #         fig[fig_name] = plt.figure(fig_name)
                #         x = loss_performance_vs_total_FLOPs_ratio[2]
                #         y = loss_performance_vs_total_FLOPs_ratio[0]
                #         yerr = loss_performance_vs_total_FLOPs_ratio[1]
                #         prune_ratio_list = prune_ratio.split('-')
                #         key_for_dict = f"{prune_ratio_list[0]}"
                #         record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Relative FLOPs ratio', y_label=flops_metric_name)
                #         print('loss_performance_vs_total_FLOPs_ratio', loss_performance_vs_total_FLOPs_ratio)
                #         # draw_macs_perform_figure(plt, x, y, yerr, key_for_dict, 'Relative FLOPs ratio', flops_metric_name, y_lim=performance_metric_max)
                #         loss_performance_vs_total_FLOPs_ratio = [None, None, None]
                
                # if 'gridsearch' in prune_method and ('Perplexity' in index or 'fullinf_FLOPs_ratio_for_all_layers' in index):
                #     if any(metric_name in index for metric_name in metric_name_list):
                #         flops_metric_name = next((metric for metric in metric_name_list if metric in index), None)
                #         flops_metric_name = flops_metric_name.split('/')[1]
                #         if ppl_performance_vs_total_FLOPs_ratio[0] is None:
                #             ppl_performance_vs_total_FLOPs_ratio[0] = row.tolist()[0]
                #             ppl_performance_vs_total_FLOPs_ratio[1] = row_se.tolist()[0]
                #     elif 'fullinf_FLOPs_ratio_for_all_layers' in index :
                #         if ppl_performance_vs_total_FLOPs_ratio[2] is None:
                #             ppl_performance_vs_total_FLOPs_ratio[2] = row.tolist()[0]

                    
                #     if ppl_performance_vs_total_FLOPs_ratio[0] is not None and ppl_performance_vs_total_FLOPs_ratio[2] is not None:
                #         print('performancevssparsity', ppl_performance_vs_total_FLOPs_ratio, flops_metric_name, prune_ratio)
                #         fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, prune_metric, prune_method, mode,\
                #         calib_info, prune_info, cust_tgt_modules, 'FIG:Perplexity_fullinf_FLOPs_ratio_for_all_layers'])
                #         fig[fig_name] = plt.figure(fig_name)
                #         x = ppl_performance_vs_total_FLOPs_ratio[2]
                #         y = ppl_performance_vs_total_FLOPs_ratio[0]
                #         yerr = ppl_performance_vs_total_FLOPs_ratio[1]
                #         prune_ratio_list = prune_ratio.split('-')
                #         key_for_dict = f"{prune_ratio_list[0]}"
                #         record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Relative FLOPs ratio', y_label=flops_metric_name)
                #         print('ppl_performance_vs_total_FLOPs_ratio', ppl_performance_vs_total_FLOPs_ratio)
                #         # draw_macs_perform_figure(plt, x, y, yerr, key_for_dict, 'Relative FLOPs ratio', flops_metric_name, y_lim=performance_metric_max)
                #         ppl_performance_vs_total_FLOPs_ratio = [None, None, None]

                if any(metric_name in index for metric_name in metric_name_list):
                    # probe generation on different size
                    if file == 'differentprobestudy':
                        metric_name = next((metric for metric in metric_name_list if metric in index), None)
                        metric_name = metric_name.split('/')[1]
                        if 'Accuracy' in metric_name:
                            metric_name = 'Accuracy'
                        print('metric_name', metric_name)
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, prune_ratio, calib_info, cust_tgt_modules, f'FIG:differentprobestudy'])
                        fig[fig_name] = plt.figure(fig_name)

                        if mode == 'asyncinter':
                            ax = fig[fig_name].add_subplot(111)
                            ax.axhline(y=row.tolist()[0], color='r', linestyle='-', linewidth=2, label=f'w/o PP')
                            continue

                        probe_seq_ratio, probe_batch_size = get_probe_type_ratio(prune_info, batch_size)
                        x = probe_seq_ratio
                        y = row.tolist()[0]
                        yerr = row_se.tolist()[0]
                        prune_ratio_list = prune_ratio.split('-')
                        key_for_dict = f"Probe Batch Size {probe_batch_size}"
                        print('key_for_dict', key_for_dict)
                        record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Probe Sequence Ratio', y_label=metric_name)
                    
                    elif file == 'ablationforfusion':
                        def extract_number(input_string):
                            # Use regular expression to find numbers in the string
                            result = re.findall(r'\d+\.\d+', input_string)
                            if result:
                                # Convert the first found number from string to float
                                return result[0]
                            else:
                                # Optionally handle the case where no number is found
                                return None
    
                        metric_name = next((metric for metric in metric_name_list if metric in index), None)
                        metric_name = metric_name.split('/')[1]
                        if 'Accuracy' in metric_name:
                            metric_name = 'Accuracy'
                        print('metric_name', metric_name)
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, prune_ratio, calib_info, cust_tgt_modules, f'FIG:ablationforfusion'])
                        fig[fig_name] = plt.figure(fig_name)

                        # if mode == 'asyncinter':
                        #     ax = fig[fig_name].add_subplot(111)
                        #     ax.axhline(y=row.tolist()[0], color='r', linestyle='-', linewidth=2, label=f'w/o PP')
                        #     continue

                        probe_seq_ratio, probe_batch_size = get_probe_type_ratio(prune_info, batch_size)
                        x = probe_seq_ratio
                        y = row.tolist()[0]
                        yerr = row_se.tolist()[0]
                        # prune_ratio_list = prune_ratio.split('-')
                        if 'probefixratio' in prune_method:
                            key_for_dict = f'Fix-{extract_number(prune_method)}'
                        else:
                            key_for_dict = 'Importance-Scaled Fusion'
                        # key_for_dict = f"Probe Batch Size {probe_batch_size}"
                        print('xyyerr', x, y, yerr)
                        print('key_for_dict', key_for_dict)
                        record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Probe Sequence Ratio', y_label=metric_name)

                        
                    else:
                        metric_name = next((metric for metric in metric_name_list if metric in index), None)
                        metric_name = metric_name.split('/')[1]
                        print('metric_name', metric_name)
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, \
                        calib_info, cust_tgt_modules, f'FIG:{metric_name}'])
                        fig[fig_name] = plt.figure(fig_name)
                        x = prune_ratio
                        y = row.tolist()[0]
                        yerr = row_se.tolist()[0]
                        prune_ratio_list = prune_ratio.split('-')
                        key_for_dict = f"{prune_method}_{mode}_{prune_info}_{metric_name}"
                        record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Prune Ratio', y_label=metric_name)

                
        #         if 'diff_ratio' in index:
        #             layer_number = find_layer_number(index)
        #             if 'attn' in index:
        #                 fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, prune_metric, prune_method, mode,\
        # calib_info, prune_info, cust_tgt_modules, f'FIG:attn_diff_ratio'])
        #             elif 'mlp' in index:
        #                 fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, prune_metric, prune_method, mode,\
        # calib_info, prune_info, cust_tgt_modules, f'FIG:mlp_diff_ratio'])
                    
        #             fig[fig_name] = plt.figure(fig_name)
        #             x = layer_number
        #             y = row.tolist()[0]
        #             yerr = row_se.tolist()[0]

        #             key_for_dict = f"Pruning Ratio {prune_ratio}"
        #             record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Layer Order', y_label='Different Channel Ratio')
                

                

                    
        #         if 'cur_select_indices' in index:
        #             layer_number = find_layer_number(index)
        #             print('prune_method', prune_method, prune_info, index)
        #             print('layer_number', layer_number)
        #             if 'attn' in index:
        #                 key = f'attn_{layer_number}'
        #             elif 'mlp' in index:
        #                 key = f'mlp_{layer_number}'

        #             if '1-1-1-1-1-seq' in prune_info:
        #                 select_channels_indices[key] = set(row.tolist())
        #                 continue
        #             else:
        #                 if 'probe' in prune_method or 'flap' in prune_method:
        #                     intersection_ratio =  (len(set(select_channels_indices[key])) - len(select_channels_indices[key].intersection(set(row.tolist())))) / len(set(select_channels_indices[key]))

        #             if 'attn' in index:
        #                 fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, prune_ratio,\
        # calib_info,  cust_tgt_modules, f'FIG:attn_same_ratio_with_upperbound'])
        #             elif 'mlp' in index:
        #                 fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, prune_ratio, \
        # calib_info,  cust_tgt_modules, f'FIG:mlp_same_ratio_with_upperbound'])

        #             fig[fig_name] = plt.figure(fig_name)
        #             x = layer_number
        #             y = intersection_ratio
        #             yerr = 0

        #             if 'probe' in prune_method:
        #                 key_for_dict = 'PP'
        #             elif 'flap' in prune_method:
        #                 key_for_dict = 'FLAP'
        #             record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Layer Order', y_label='Different Channel Ratio')
        #             pass

                def mean_indices_common_ratio(upperbound, method):
                    return_list = []
                    for i in range(len(upperbound)):
                        cur_upperbound = upperbound[i]
                        cur_method = method[i]
                        print('cur_upperbound', cur_upperbound)
                        print('cur_method', cur_method)
                        ratio = len(set(cur_upperbound).intersection(set(cur_method))) / len(set(cur_upperbound).union(set(cur_method)))
                        return_list.append(ratio)
                    
                    mean = sum(return_list) / len(return_list)
                    se = np.std(return_list) / np.sqrt(len(return_list))
                    return mean, se
                
                if 'cur_select_indices' in index:
                    row = row[0]

                    layer_number = find_layer_number(index)
                    print('prune_method', prune_method, prune_info, index)
                    print('layer_number', layer_number)
                    if 'attn' in index:
                        key = f'attn_{layer_number}'
                    elif 'mlp' in index:
                        key = f'mlp_{layer_number}'

                    if '1-1-1-1-1-seq' in prune_info:
                        # TODO: hardcode, we only run 1 seed
                        print('len(row)', len(row.tolist()), row.tolist())
                        select_channels_indices[key] = row.tolist()
                        continue
                    else:
                        prune_method_list = prune_method.split('-')
                        print('prune_method_list', prune_method_list)
                        if len(prune_method_list) == 2 and prune_method_list[0] == 'calib':
                            # intersection_ratio = len(set(select_channels_indices[key]).intersection(set(row.tolist()))) / len(set(select_channels_indices[key]).union(set(row.tolist())))

                            y, yerr = mean_indices_common_ratio = mean_indices_common_ratio(select_channels_indices[key], row.tolist())
                            key_for_dict = f"w/o PP"
                        elif 'probe' in prune_method or 'flap' in prune_method:
                            # intersection_ratio = len(set(select_channels_indices[key]).intersection(set(row.tolist()))) / len(set(select_channels_indices[key]))
                            # intersection_ratio = len(set(select_channels_indices[key]).intersection(set(row.tolist()))) / len(set(select_channels_indices[key]).union(set(row.tolist())))
                            y, yerr = mean_indices_common_ratio = mean_indices_common_ratio(select_channels_indices[key], row.tolist())
                            key_for_dict = f"w/ PP"

                    if 'attn' in index:
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_metric, seq_len, calib_info,  cust_tgt_modules, f'FIG:attn_same_ratio_with_upperbound'])
                    elif 'mlp' in index:
                        fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_metric, seq_len, calib_info,  cust_tgt_modules, f'FIG:mlp_same_ratio_with_upperbound'])

                    fig[fig_name] = plt.figure(fig_name)
                    x = layer_number
                    # y = intersection_ratio
                    # yerr = 0

                    # key_for_dict = f"Pruning Ratio {prune_ratio}"
                    record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Layer Order', y_label='Common Pruning Channels Jaccard Index')
                    pass
                
                if 'inference_duration' in index:
                    layer_number = find_layer_number(index)
                    if 'attn' in index:
                        name = 'attn_inference_duration'
                    else:
                        name = 'mlp_inference_duration'
                    fig_name = '_'.join([data_name, model_name, task_name, prune_ratio, batch_size, seq_len, calib_info, cust_tgt_modules, f'FIG:{name}'])
                    fig[fig_name] = plt.figure(fig_name)
                    x = layer_number
                    y = row.tolist()[0]
                    yerr = row_se.tolist()[0]
                    # prune_ratio_list = prune_ratio.split('-')
                    key_for_dict = prune_method
                    record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Prune Ratio', y_label='ms')
                
        
    for fig_name in fig_data_across_multi_indices:
        print('fig_name', fig_name)
        fig[fig_name] = plt.figure(fig_name)
        for key_for_dict in fig_data_across_multi_indices[fig_name]:
            x = fig_data_across_multi_indices[fig_name][key_for_dict]['x']
            y = fig_data_across_multi_indices[fig_name][key_for_dict]['y']
            yerr = fig_data_across_multi_indices[fig_name][key_for_dict]['yerr']
            x_label = fig_data_across_multi_indices[fig_name][key_for_dict]['x_label'][0]
            y_label = fig_data_across_multi_indices[fig_name][key_for_dict]['y_label'][0]

            if 'Loss' in fig_name or 'Perplexity' in fig_name or 'CsrAccuracyNorm' in fig_name or 'CsrAccuracy' in fig_name:
                # draw_macs_perform_figure(plt, x, y, yerr, key_for_dict, x_label, y_label, y_lim=performance_metric_max)
                draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label, y_label)
            # if 'diff_ratio' in fig_name:
            #     draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label, y_label)
            if 'same_ratio_with_upperbound' in fig_name:
                draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label, y_label, label_loc='lower right', x_label_font_size=25, y_label_font_size=15, legend_size=25)
            if 'differentprobestudy' in fig_name:
                draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label, y_label, label_loc='center right', x_label_font_size=25, y_label_font_size=25)
            if 'inference_duration' in fig_name:
                draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label, y_label)
            if 'ablationforfusion' in fig_name:
                draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label, y_label, label_loc='upper right', x_label_font_size=25, y_label_font_size=25)
            # if 'all_methods_performance_vs_FLOPs_ratio_for_all_layers' in fig_name:
            #     draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label, y_label)
            # if 'time_cost_per_sample' in fig_name:
            #     draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label, y_label)
            # if 'all_methods_performance_vs_prune_ratio_for_all_layers' in fig_name:
            #     print('all_methods_performance_vs_prune_ratio_for_all_layers', fig_name)
            #     print('x', x)
            #     print('y', y)
            #     draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label, y_label)
            # if 'mean_intersection_ratio' in fig_name:
            #     draw_str_x_figure(plt, x, y, yerr, key_for_dict, x_label, y_label)
            # if 'diff_intersection_ratio' in fig_name:
            #     draw_str_x_random_color_figure(plt, x, y, yerr, key_for_dict, x_label, y_label)



    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        # plt.figure(figsize=(10, 8))
        fig_name_list = fig_name.split('_')
        FIG_NAME = fig_name.split('FIG:')[-1]
        data_name = fig_name_list[0]
        model_name = fig_name_list[1]
        vis_path = os.path.join('output', 'vis', '{}'.format(save_format), args['file'], data_name, model_name, FIG_NAME)
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()









    
# def is_valid_layer_for_detailed_info(index, model_name):
#     if not plot_layer_detail:
#         return False
    
#     if 'global' in index:
#         return True

#     if model_name in total_layers:
#         # layer_number = int(index.split(".layers.")[1].split(".")[0])
#         layer_number = index.split('/')[1]

#         # Find all numbers in the string
#         layer_number = re.findall(r'\d+', layer_number)

#         # Join the number strings and convert to an integer
#         layer_number = int(''.join(layer_number)) if layer_number else None
#         if layer_number <= math.ceil(total_layers[model_name] * 0.05) or math.ceil(layer_number >= total_layers[model_name] * 0.95):
#             # print(f'layer_number: {layer_number}')
#             return True
#     # print(f'False layer_number: {layer_number}')
#         return False
#     else:
#         return True
    
# def draw_str_x_random_color_figure(plt, x, y, yerr, key_for_dict, x_label='Activation Layers in Order', y_label='Accuracy'):
#     fig_color = random.choice(list(backup_color_set))
#     fig_linestyle = random.choice(list(backup_linestyle_set))
#     fig_marker = random.choice(list(backup_marker_set))
#     # if key_for_dict not in color:
#     #     color[key_for_dict] = fig_color
#     #     linestyle[key_for_dict] = fig_linestyle
#     #     marker[key_for_dict] = fig_marker

#     #     backup_color_set.remove(fig_color)
#     #     backup_linestyle_set.remove(fig_linestyle)
#     #     backup_marker_set.remove(fig_marker)
#     if label_exists(plt, key_for_dict):
#         print('label exists', key_for_dict)
#         return
#         # plt.scatter(x, y, color=fig_color, linestyle=fig_linestyle)
#     # else:
#     #     plt.scatter(x, y, color=fig_color, linestyle=fig_linestyle, label=key_for_dict)
#         # plt.scatter(x, y, color=fig_color, linestyle=fig_linestyle)
#     plt.plot(x, y, color=fig_color, linestyle=fig_linestyle, marker=fig_marker, label=key_for_dict)
#     # plt.fill_between(x, (y - yerr), (y + yerr), color=color[algo_mode], alpha=.1)
#     plt.errorbar(x, y, yerr=yerr, color=fig_color, linestyle=fig_linestyle)
#     plt.xlabel(x_label, fontsize=fontsize['label'])
#     plt.ylabel(y_label, fontsize=fontsize['label'])
#     plt.xticks(fontsize=fontsize['ticks'])
#     plt.yticks(fontsize=fontsize['ticks'])
#     plt.legend(loc=loc_dict['label'], fontsize=fontsize['legend'], bbox_to_anchor=(1, 0.5))
#     return

# def draw_3d_heatmap(plt, fig, x, x_label='Activation Layers in Order', y_label='Accuracy', z_label='Accuracy', index=None):
#     granuality = 3000
#     dimension = len(x)
#     print('len(x)', len(x), x[-1000:])
#     pace = int(len(x) // granuality)
#     simplified_input_data = [x[i] for i in range(pace, dimension, pace)]
#     if simplified_input_data[-1] != x[-1]:
#         simplified_input_data.append(x[-1])
#     simplified_input_data = np.array(simplified_input_data)
#     x = np.array(list(range(len(simplified_input_data)+1)))
#     y = np.array(list(range(len(simplified_input_data)+1)))
#     x, y = np.meshgrid(x, y)
#     eta = np.full(x.shape, np.nan)
#     mask = y < x

#     # Applying the mask
#     x = np.where(mask, x, np.nan)  # Replace values not in the upper triangle with NaN
#     y = np.where(mask, y, np.nan)

#     pq_p = 1
#     pq_q = 2

#     # print(len(x), len(x[0]))
#     for d in range(1, len(x)):
#         # m at most equals to d-1
#         cur_dimension = min(d * pace, dimension)
#         pq_index = simplified_input_data[d-1]
#         for m in range(1, d):
#             cur_rest_dimension = m * pace

#             sub_eta = ((cur_rest_dimension / (((1 - pq_index) ** (pq_q * pq_p / (pq_q - pq_p))) * cur_dimension)) ** (-(pq_q - pq_p) / pq_q)) - 1
#             lower_bound = cur_dimension * (1 + 0) ** (-pq_q / (pq_q - pq_p)) * ((1 - pq_index) ** (pq_q * pq_p / (pq_q - pq_p)))
#             if sub_eta < 0:
#                 sub_eta = -1
#             elif sub_eta > 2:
#                 sub_eta = 2
#             # if d > 3400:
#             eta[m][d] = sub_eta
#             # if d > 3665 and m < 1000:
#             #     print(d, m, cur_dimension, cur_rest_dimension, 'eta', eta[m][d], 'lower_bound', lower_bound, lower_bound/pace, pq_index)

#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(x, y, eta, cmap='viridis')
#     fig.colorbar(surf, shrink=0.5, aspect=5)

#     elev = 18  # Elevation angle in degrees
#     azim = 45  # Azimuth angle in degrees
#     ax.view_init(elev=elev, azim=azim)
#     ax.set_title('3D Heatmap')
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     ax.set_zlabel(z_label)
#     # plt.show()
#     return




# def draw_histogram(plt, data, bins=500, density=False, x_label='Value', y_label='Frequency', title='Data Distribution'):
#     plt.hist(data, bins=bins, density=density, color='blue', edgecolor='black')
#     plt.xlabel(x_label, fontsize=fontsize['label'])
#     plt.ylabel(y_label, fontsize=fontsize['label'])
#     # plt.title(title, fontsize=14)
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
#     return

# def process_distri(data, left_range=-1, right_range=1):
#     bin_edges = [
#         -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, # -1000 to -100
#         -90, -80, -70, -60, -50, -40, -30, -20, -10,  
#         10, 20, 30, 40, 50, 60, 70, 80, 90, 100,  # -100 to 100 
#         200, 300, 400, 500, 600, 700, 800, 900, 1000  # 100 to 1000
#     ]
#     fine_bins = np.arange(-10, 10, 0.1).tolist()
#     # for drawing 0
#     bin_edges = bin_edges + fine_bins + [0.01]
#     bin_edges = sorted(set(bin_edges))

#     # Find the index for left_range and right_range
#     left_index = next(i for i, x in enumerate(bin_edges) if x >= left_range)
#     right_index = next(i for i, x in enumerate(bin_edges) if x >= right_range)
#     return data[left_index:right_index+1], bin_edges[left_index:right_index+2]

# def cal_prune_count_base_on_pq(sorted_tensor, pq_p, pq_q, eta, pq_beta, pq_gamma, return_norm='p'):

#     # norm_across_other_dims = norm_across_other_dims + (norm_across_other_dims == 0) * 1e-9
#     # Calculate norms only for non-zero channels
#     # non_zero_norms = norm_across_other_dims[non_zero_mask]
#     norm_p = torch.linalg.vector_norm(sorted_tensor, ord=pq_p, dim=0)
#     norm_q = torch.linalg.vector_norm(sorted_tensor, ord=pq_q, dim=0) + 1e-10
    
#     dimension = sorted_tensor.shape[0]
#     pq_indices = (1 - dimension ** (1/pq_q - 1/pq_p) * (norm_p / norm_q))
    
#     # add additional dimension if dimension is 0
#     # if pq_indices.dim() == 0 or pq_indices.dim() == 1:
#     #     pq_indices.unsqueeze_(0)
#     print('pq_indices', pq_indices, dimension)
#     if torch.isnan(pq_indices).any() or torch.isinf(pq_indices).any():
#         # pq_indices = torch.min(pq_indices, torch.ones_like(pq_indices))
#         pq_indices = torch.tensor(1)
#         # raise ValueError('pq_indices contains nan values')

#     lower_bound = dimension * (1 + eta) ** (-pq_q / (pq_q - pq_p)) * ((1 - pq_indices) ** (pq_q * pq_p / (pq_q - pq_p)))
#     print('lower_bound', lower_bound, dimension)
#     beta_tensor = torch.full_like(lower_bound, pq_beta)
#     prune_channels_count = torch.floor(dimension * torch.min(pq_gamma * (1 - lower_bound / dimension), beta_tensor))
#     print('prune_channels_count', prune_channels_count)
#     if return_norm == 'p':
#         return int(lower_bound), pq_indices, norm_p
#     elif return_norm == 'q':
#         return int(lower_bound), pq_indices, norm_q



# # several methods for all layers on 1 plot
                    # if any(metric_name in index for metric_name in metric_name_list) or 'FLOPs_ratio_for_pruned_layers' in index:
                    #     if any(metric_name in index for metric_name in metric_name_list):
                    #         flops_metric_name = next((metric for metric in metric_name_list if metric in index), None)
                    #         flops_metric_name = flops_metric_name.split('/')[1]
                    #         if performance_vs_prunedflops[0] is None:
                    #             performance_vs_prunedflops[0] = min(performance_metric_max, row.tolist()[0])
                    #             performance_vs_prunedflops[1] = min(performance_metric_max, row_se.tolist()[0])
                    #     elif 'FLOPs_ratio_for_pruned_layers' in index or 'FLOPs_for_pruned_layers' in index:
                    #         if performance_vs_prunedflops[2] is None:
                    #             performance_vs_prunedflops[2] = row.tolist()[0]
                        
                    #     if performance_vs_prunedflops[0] is not None and performance_vs_prunedflops[2] is not None:
                    #         print('performancevssparsity', performance_vs_prunedflops, flops_metric_name, prune_ratio)
                    #         fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, cust_tgt_modules, 'FIG:all_methods_performance_vs_FLOPs_ratio_for_pruned_layers'])
                    #         fig[fig_name] = plt.figure(fig_name)
                    #         x = performance_vs_prunedflops[2]
                    #         y = performance_vs_prunedflops[0]
                    #         yerr = performance_vs_prunedflops[1]
                    #         # if 'pq' in prune_name and 'WIFV' in prune_metric:
                    #         #     prune_name += '-flap'
                    #         # elif 'O1WIFN' in prune_metric or 'O2WIFN' in prune_metric:
                    #         #     prune_name += prune_metric
                    #         # elif 'pq' in prune_name and 'WIFN' in prune_metric:
                    #         #     prune_name += '-wanda'
                    #         # elif 'IFN' in prune_metric:
                    #         #     prune_name += prune_metric
                    #         key_for_dict = f"{prune_name}"
                    #         # if 'pq' in prune_name:
                    #         #     key_for_dict = f"Our"
                    #         # elif 'mag' in prune_name:
                    #         #     # print('prune_ratio', prune_ratio, prune_ratio==0, type(prune_ratio))
                    #         #     if float(prune_ratio) == 0:
                    #         #         key_for_dict = f"Dense"
                    #         #     else:
                    #         #         key_for_dict = f"Mag"
                    #         record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Relative FLOPs ratio', y_label=flops_metric_name)
                    #         # draw_macs_perform_figure(plt, x, y, yerr, key_for_dict, 'Relative FLOPs ratio', flops_metric_name, y_lim=performance_metric_max)
                    #         performance_vs_prunedflops = [None, None, None]

                    # if any(metric_name in index for metric_name in metric_name_list) or 'FLOPs_ratio_for_all_layers' in index:
                    #     if any(metric_name in index for metric_name in metric_name_list):
                    #         flops_metric_name = next((metric for metric in metric_name_list if metric in index), None)
                    #         flops_metric_name = flops_metric_name.split('/')[1]
                    #         if performance_vs_total_FLOPs_ratio[0] is None:
                    #             performance_vs_total_FLOPs_ratio[0] = min(performance_metric_max, row.tolist()[0])
                    #             performance_vs_total_FLOPs_ratio[1] = min(performance_metric_max, row_se.tolist()[0])
                    #     elif 'FLOPs_ratio_for_all_layers' in index:
                    #         if performance_vs_total_FLOPs_ratio[2] is None:
                    #             performance_vs_total_FLOPs_ratio[2] = row.tolist()[0]
                        
                    #     if performance_vs_total_FLOPs_ratio[0] is not None and performance_vs_total_FLOPs_ratio[2] is not None:
                    #         print('performancevssparsity', performance_vs_total_FLOPs_ratio, flops_metric_name, prune_ratio)
                    #         fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, cust_tgt_modules, 'FIG:all_methods_performance_vs_FLOPs_ratio_for_all_layers'])
                    #         fig[fig_name] = plt.figure(fig_name)
                    #         x = performance_vs_total_FLOPs_ratio[2]
                            
                    #         y = performance_vs_total_FLOPs_ratio[0]
                    #         yerr = performance_vs_total_FLOPs_ratio[1]
                    #         # if 'pq' in prune_name and 'WIFV' in prune_metric:
                    #         #     prune_name += '-flap'
                    #         # elif 'O1WIFN' in prune_metric or 'O2WIFN' in prune_metric:
                    #         #     prune_name += prune_metric
                    #         # elif 'pq' in prune_name and 'WIFN' in prune_metric:
                    #         #     prune_name += '-wanda'
                    #         # elif 'IFN' in prune_metric:
                    #         #     prune_name += prune_metric
                    #         key_for_dict = f"{prune_name}"
                            
                    #         # if 'pq' in prune_name:
                    #         #     key_for_dict = f"Our"
                    #         # elif 'mag' in prune_name:
                    #         #     # print('prune_ratio', prune_ratio, prune_ratio==0, type(prune_ratio))
                    #         #     if float(prune_ratio) == 0:
                    #         #         key_for_dict = f"Dense"
                    #         #     else:
                    #         #         key_for_dict = f"Mag"
                    #         record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Relative FLOPs ratio', y_label=flops_metric_name)
                    #         # draw_macs_perform_figure(plt, x, y, yerr, key_for_dict, 'Relative FLOPs ratio', flops_metric_name, y_lim=performance_metric_max)
                    #         performance_vs_prunedflops = [None, None, None]

                    # if any(metric_name in index for metric_name in metric_name_list):
                    #     if any(metric_name in index for metric_name in metric_name_list):
                    #         flops_metric_name = next((metric for metric in metric_name_list if metric in index), None)
                    #         flops_metric_name = flops_metric_name.split('/')[1]
                                           
                    #         print('performancevssparsity', performance_vs_total_FLOPs_ratio, flops_metric_name, prune_ratio)
                    #         fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, cust_tgt_modules, 'FIG:all_methods_performance_vs_prune_ratio_for_all_layers'])
                    #         fig[fig_name] = plt.figure(fig_name)
                    #         x = float(prune_ratio)
                            
                    #         y = min(performance_metric_max, row.tolist()[0])
                    #         yerr = min(performance_metric_max, row_se.tolist()[0])
                    #         # if 'pq' in prune_name and 'WIFV' in prune_metric:
                    #         #     prune_name += '-flap'
                    #         # elif 'O1WIFN' in prune_metric or 'O2WIFN' in prune_metric:
                    #         #     prune_name += prune_metric
                    #         # elif 'pq' in prune_name and 'WIFN' in prune_metric:
                    #         #     prune_name += '-wanda'
                    #         # elif 'IFN' in prune_metric:
                    #         #     prune_name += prune_metric
                    #         key_for_dict = f"{prune_name}"
                    #         record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Pruned ratio', y_label=flops_metric_name)
                    #         if batch_size == '1':
                    #             temp_list = ['10']
                    #             for item in temp_list:
                    #                 fig_name = '_'.join([data_name, model_name, task_name, item, seq_len, cust_tgt_modules, 'FIG:all_methods_performance_vs_prune_ratio_for_all_layers'])
                    #                 key_for_dict = 'bsz1_' + key_for_dict
                    #                 record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Pruned ratio', y_label=flops_metric_name)
                    #         # if 'pq' in prune_name:
                    #         #     key_for_dict = f"Our"
                    #         # elif 'mag' in prune_name:
                    #         #     # print('prune_ratio', prune_ratio, prune_ratio==0, type(prune_ratio))
                    #         #     if float(prune_ratio) == 0:
                    #         #         key_for_dict = f"Dense"
                    #         #     else:
                    #         #         key_for_dict = f"Mag"
                    
    
                    # if 'position_distribution_1' in index:
                    #     layer_number = index.split('/')[1]
                    #     fig_name = '_'.join([data_name, model_name, task_name, batch_size, layer_number, prune_ratio, seq_len, cust_tgt_modules,'FIG:', 'position_distribution'])
                    #     fig[fig_name] = plt.figure(fig_name)
                    #     # x = fullinf_vs_optimal_select_mean_intersection_ratio_order
                    #     # cur_bsz_mean_intersection_ratio_order += 1
                        
                    #     y = row.tolist()
                    #     yerr = row_se.tolist()
                    #     if 'position_distribution_1' in index:
                    #         print('\nposition_distribution_1', len(y), layer_number)
                    #     key_for_dict = prune_name
                    #     draw_histogram(plt, y, bins=2000, density=True, x_label='Value', y_label='Nums', title='Position Distribution')


                    # if 'fullinf_vs_optimal_select_mean_intersection_ratio' in index:
                    #     fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_ratio, seq_len, cust_tgt_modules,'FIG:', 'fullinf_vs_optimal_select_mean_intersection_ratio'])
                    #     fig[fig_name] = plt.figure(fig_name)
                    #     x = fullinf_vs_optimal_select_mean_intersection_ratio_order
                    #     # cur_bsz_mean_intersection_ratio_order += 1
                    #     y = row.tolist()[0]
                    #     yerr = row_se.tolist()[0]
                    #     key_for_dict = prune_name
                    #     record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Layer order', y_label='Ratio')

                    # if 'probe_vs_optimal_select_mean_intersection_ratio' in index:
                    #     fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_ratio, seq_len, cust_tgt_modules,'FIG:', 'probe_vs_optimal_select_mean_intersection_ratio'])
                    #     fig[fig_name] = plt.figure(fig_name)
                    #     x = probe_vs_optimal_select_mean_intersection_ratio_order
                    #     # probe_mean_intersection_ratio += 1
                    #     y = row.tolist()[0]
                    #     yerr = row_se.tolist()[0]
                    #     # print(x,y)
                    #     key_for_dict = prune_name
                    #     record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Layer order', y_label='Ratio')
                    
                    # if 'probe_vs_fullinf_select_mean_intersection_ratio' in index:
                    #     fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_ratio, seq_len, cust_tgt_modules,'FIG:', 'probe_vs_fullinf_select_mean_intersection_ratio'])
                    #     fig[fig_name] = plt.figure(fig_name)
                    #     x = probe_vs_fullinf_select_mean_intersection_ratio_order
                    #     # probe_mean_intersection_ratio += 1
                    #     y = row.tolist()[0]
                    #     yerr = row_se.tolist()[0]
                    #     # print(x,y)
                    #     key_for_dict = prune_name
                    #     record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Layer order', y_label='Ratio')

                    # if 'fullinf_vs_optimal_select_mean_intersection_ratio' in index or 'probe_vs_optimal_select_mean_intersection_ratio' in index or 'probe_vs_fullinf_select_mean_intersection_ratio' in index:
                    #     fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_ratio, seq_len, prune_name, cust_tgt_modules,'FIG:', 'comparison_for_select_diff_intersection_ratio'])
                    #     fig[fig_name] = plt.figure(fig_name)
                    #     if 'fullinf_vs_optimal_select_mean_intersection_ratio' in index:
                    #         x = fullinf_vs_optimal_select_mean_intersection_ratio_order
                    #         fullinf_vs_optimal_select_mean_intersection_ratio_order += 1
                    #         key_for_dict = f'fullinf_vs_optimal_select_{prune_name}'
                    #     elif 'probe_vs_optimal_select_mean_intersection_ratio' in index:
                    #         x = probe_vs_optimal_select_mean_intersection_ratio_order
                    #         probe_vs_optimal_select_mean_intersection_ratio_order += 1
                    #         key_for_dict = f'probe_vs_optimal_select_{prune_name}'
                    #     elif 'probe_vs_fullinf_select_mean_intersection_ratio' in index:
                    #         x = probe_vs_fullinf_select_mean_intersection_ratio_order
                    #         probe_vs_fullinf_select_mean_intersection_ratio_order += 1
                    #         key_for_dict = f'probe_vs_fullinf_select_{prune_name}'
                    #     y = row.tolist()[0]
                    #     yerr = row_se.tolist()[0]                        
                    #     record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Layer order', y_label='Ratio')

                    # if 'fullinf_vs_optimal_prune_mean_intersection_ratio' in index:
                    #     fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_ratio, seq_len, cust_tgt_modules,'FIG:', 'fullinf_vs_optimal_prune_mean_intersection_ratio'])
                    #     fig[fig_name] = plt.figure(fig_name)
                    #     x = fullinf_vs_optimal_prune_mean_intersection_ratio_order
                    #     # cur_bsz_mean_intersection_ratio_order += 1
                    #     y = row.tolist()[0]
                    #     yerr = row_se.tolist()[0]
                    #     key_for_dict = prune_name
                    #     record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Layer order', y_label='Ratio')

                    # if 'probe_vs_optimal_prune_mean_intersection_ratio' in index:
                    #     fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_ratio, seq_len, cust_tgt_modules,'FIG:', 'probe_vs_optimal_prune_mean_intersection_ratio'])
                    #     fig[fig_name] = plt.figure(fig_name)
                    #     x = probe_vs_optimal_prune_mean_intersection_ratio_order
                    #     # probe_mean_intersection_ratio += 1
                    #     y = row.tolist()[0]
                    #     yerr = row_se.tolist()[0]
                    #     # print(x,y)
                    #     key_for_dict = prune_name
                    #     record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Layer order', y_label='Ratio')
                    
                    # if 'probe_vs_fullinf_prune_mean_intersection_ratio' in index:
                    #     fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_ratio, seq_len, cust_tgt_modules,'FIG:', 'probe_vs_fullinf_prune_mean_intersection_ratio'])
                    #     fig[fig_name] = plt.figure(fig_name)
                    #     x = probe_vs_fullinf_prune_mean_intersection_ratio_order
                    #     # probe_mean_intersection_ratio += 1
                    #     y = row.tolist()[0]
                    #     yerr = row_se.tolist()[0]
                    #     # print(x,y)
                    #     key_for_dict = prune_name
                    #     record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Layer order', y_label='Ratio')

                    # if 'fullinf_vs_optimal_prune_mean_intersection_ratio' in index or 'probe_vs_optimal_prune_mean_intersection_ratio' in index or 'probe_vs_fullinf_prune_mean_intersection_ratio' in index:
                    #     fig_name = '_'.join([data_name, model_name, task_name, batch_size, prune_ratio, seq_len, prune_name, cust_tgt_modules,'FIG:', 'comparison_for_prune_diff_intersection_ratio'])
                    #     fig[fig_name] = plt.figure(fig_name)
                    #     if 'fullinf_vs_optimal_prune_mean_intersection_ratio' in index:
                    #         x = fullinf_vs_optimal_prune_mean_intersection_ratio_order
                    #         fullinf_vs_optimal_prune_mean_intersection_ratio_order += 1
                    #         key_for_dict = f'fullinf_vs_optimal_prune_{prune_name}'
                    #     elif 'probe_vs_optimal_prune_mean_intersection_ratio' in index:
                    #         x = probe_vs_optimal_prune_mean_intersection_ratio_order
                    #         probe_vs_optimal_prune_mean_intersection_ratio_order += 1
                    #         key_for_dict = f'probe_vs_optimal_prune_{prune_name}'
                    #     elif 'probe_vs_fullinf_prune_mean_intersection_ratio' in index:
                    #         x = probe_vs_fullinf_prune_mean_intersection_ratio_order
                    #         probe_vs_fullinf_prune_mean_intersection_ratio_order += 1
                    #         key_for_dict = f'probe_vs_fullinf_prune_{prune_name}'
                    #     y = row.tolist()[0]
                    #     yerr = row_se.tolist()[0]                        
                    #     record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Layer order', y_label='Ratio')

                    # if 'dense_duration_per_sample' or 'FLOPs_ratio_for_all_layers' in index:
                    #     if 'dense_duration_per_sample' in index:
                    #         dense_time_vs_total_FLOPs_ratio[0] = row.tolist()[0]
                    #         dense_time_vs_total_FLOPs_ratio[1] = row_se.tolist()[0]
                    #     elif 'FLOPs_ratio_for_all_layers' in index:
                    #         dense_time_vs_total_FLOPs_ratio[2] = row.tolist()[0]
                        
                    #     if dense_time_vs_total_FLOPs_ratio[0] is not None and dense_time_vs_total_FLOPs_ratio[2] is not None:
                    #         fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, cust_tgt_modules,'FIG:', 'time_cost_per_sample'])
                    #         fig[fig_name] = plt.figure(fig_name)
                    #         x = dense_time_vs_total_FLOPs_ratio[2]
                    #         y = dense_time_vs_total_FLOPs_ratio[0]
                    #         yerr = dense_time_vs_total_FLOPs_ratio[1]
                            
                    #         key_for_dict = "dense"
                    #         x = 1
                    #         record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Relative FLOPs ratio', y_label='Time (s)')
                    #         dense_time_vs_total_FLOPs_ratio = [None, None, None]
                    
                    # if 'pruned_duration_per_sample' in index or 'FLOPs_ratio_for_all_layers' in index:
                    #     if 'pruned_duration_per_sample' in index:
                    #         prune_time_vs_total_FLOPs_ratio[0] = row.tolist()[0]
                    #         prune_time_vs_total_FLOPs_ratio[1] = row_se.tolist()[0]
                    #     elif 'FLOPs_ratio_for_all_layers' in index:
                    #         prune_time_vs_total_FLOPs_ratio[2] = row.tolist()[0]
                        
                    #     if prune_time_vs_total_FLOPs_ratio[0] is not None and prune_time_vs_total_FLOPs_ratio[2] is not None:
                    #         fig_name = '_'.join([data_name, model_name, task_name, batch_size, seq_len, cust_tgt_modules,'FIG:', 'time_cost_per_sample'])
                    #         fig[fig_name] = plt.figure(fig_name)
                    #         x = prune_time_vs_total_FLOPs_ratio[2]
                    #         y = prune_time_vs_total_FLOPs_ratio[0]
                    #         yerr = prune_time_vs_total_FLOPs_ratio[1]
                            
                    #         key_for_dict = f"{prune_name}"
                    #         record_fig_data_across_multi_indices(fig_data_across_multi_indices, fig_name, key_for_dict, x=x, y=y, yerr=yerr, x_label='Relative FLOPs ratio', y_label='Time (s)')
                    #         prune_time_vs_total_FLOPs_ratio = [None, None, None]

