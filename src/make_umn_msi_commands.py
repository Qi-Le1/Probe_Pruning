import argparse
import itertools
import os 
import errno

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--pbs_prefix', default=None, type=str)
parser.add_argument('--code_folder', default=None, type=str)
parser.add_argument('--res_folder', default=None, type=str)
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--file', default=None, type=str)
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--log_interval', default=None, type=float)
args = vars(parser.parse_args())

def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return

def make_controls(script_name, init_seeds, device, resume_mode, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + device + resume_mode + init_seeds + control_names 
    controls = list(itertools.product(*controls))
    # print('---controls', controls)
    return controls

'''
run: train or test
init_seed: 0
world_size: 1
num_experiments: 1
resume_mode: 0
log_interval: 0.25
num_gpus: 12
round: 1
experiment_step: 1
file: train_后面的, 例如privacy_joint
data: ML100K_ML1M_ML10M_ML20M

python create_commands_for_large_scale.py --run train --num_gpus 4 --round 1 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data ML100K_ML1M_ML10M_ML20M

control:
  data_name: CIFAR10 (Name of the dataset)
  model_name: resnet9
  num_clients: 100
  data_split_mode: iid / non-iid
  algo_mode: dynamicsgd / fedavg / dynamicfl / dynamicsgd / fedgen / fenensemble / FedProx
  select_client_mode: fix / dynamic
  client_ratio: 0.2-0.3-0.5
  number_of_freq_levels: 2-3-4
  max_local_gradient_update: 50
  
# experiment
num_workers: 0
init_seed: 0
num_experiments: 1
log_interval: 0.25
device: cuda
resume_mode: 0
verbose: False
'''

# test dynamicsgd / fedavg / dynamicfl / dynamicsgd / fedgen / fenensemble / FedProx
# test num_clients: 100 / 300
# test datasets: CIFAR10 / CIFAR100 / FEMNIST
# test data_split_mode: iid / non-iid
# test max_local_gradient_update: 10 / 50 / 100
# for dynamicfl: test select_client_mode: fix / dynamic
# for dynamicfl: test client_ratio: 0.2-0.3-0.5
# for dynamicfl: test number_of_freq_levels: 2-3-4

def main():
    pbs_prefix = args['pbs_prefix']
    code_folder = args['code_folder']
    res_folder = args['res_folder']
    run = args['run']
    num_gpus = args['num_gpus']
    round = args['round']
    world_size = args['world_size']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    temp_num_experiments = num_experiments
    resume_mode = args['resume_mode']
    log_interval = args['log_interval']
    device = args['device']
    file = args['file']
    data = args['data'].split('_')
    
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    log_interval = [[log_interval]]
    device = [[device]]
    filename = '{}'.format(file)


    
    if file == 'test_model' or file == 'test_local_tuned_model':
        controls = []
        script_name = [[f'{filename}.py']]
        if 's2s' in data:
            control_name = [[['fpb-sa', 'wikisql'], ['bart-base'], ['s2s'], ['100'], ['pq-h-2-0-2-max'], 
                    ['inter'], ['somemethods-3']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
        elif 'sc' in data:
            # control_name = [[['glue-cola', 'glue-mnli', 'glue-mrpc', 'glue-qnli', 'glue-qqp', 'glue-rte', 'glue-sst2', 'glue-stsb'], ['roberta-base'], ['sc'], ['100'], ['pq-h-2-0-2-max'], 
            #         ['inter'], ['somemethods-3'], ['query-value']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['glue-cola', 'glue-mnli'], ['roberta-base'], ['sc'], ['10', '100'], ['pq-h-2-0-2-max'], 
            #         ['inter'], ['somemethods-3'], ['query-value']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # control_name = [[['glue-cola', 'glue-mnli'], ['roberta-base'], ['sc'], ['10', '100'], ['pq-h-2-0-2-max'], 
            #         ['inter'], ['somemethods-3'], ['attention.output.dense', 'intermediate.dense', 'output.dense', 'dense']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['glue-cola', 'glue-mnli', 'glue-mrpc', 'glue-qnli', 'glue-qqp', 'glue-rte', 'glue-sst2', 'glue-stsb'], ['roberta-base'], ['sc'], ['1'], ['pqstruct-h-2-0.05-2-max'], 
            #         ['inter'], ['somemethods-3'], ['output.dense']]]
            # control_name = [[['glue-cola', 'glue-mnli', 'glue-mrpc', 'glue-qnli', 'glue-qqp', 'glue-rte', 'glue-sst2', 'glue-stsb'], ['roberta-base'], ['sc'], ['1'], [f'pqstruct-h-2-{x}-2-max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3]], 
            #         ['inter'], ['somemethods-3'], ['output.dense']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            control_name = [[['glue-cola', 'glue-mnli', 'glue-mrpc', 'glue-qnli', 'glue-qqp', 'glue-rte', 'glue-sst2', 'glue-stsb'], ['roberta-base'], ['sc'], ['1'], [f'pqstruct-h-2-{x}-2-max' for x in [9999]], 
                    ['inter'], ['somemethods-3'], ['output.dense']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
        elif 'clm' in data:
            # ------- llama-2-7b
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['3'], [f'magunstructglobal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]],
            #         ['full'], ['somemethods-3'], ['down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['3'], [f'magstructglobal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]],
            #         ['full'], ['somemethods-3'], ['down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'magstructlocal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]],
            #         ['full'], ['somemethods-3'], ['down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
            #         ['inter'], ['somemethods-3'], ['gate-proj', 'up-proj', 'down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #         ['inter'], ['somemethods-3'], ['down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'w*pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #         ['inter'], ['somemethods-3'], ['down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [9999]],
            #         ['inter'], ['somemethods-3'], ['default', 'gate-proj+up-proj+down-proj', 'down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'w*pqstructlocal+h+2+{x}+-1+max' for x in [9999]],
            #         ['inter'], ['somemethods-3'], ['default', 'gate-proj+up-proj+down-proj', 'down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
            #         ['inter'], ['somemethods-3'], ['gate-proj', 'up-proj', 'down-proj', 'gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
            #         ['inter'], ['somemethods-3'], ['o-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
            #         ['inter'], ['somemethods-3'], ['o-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
            #         ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['2048'], ['l2'], [f'pqstructlocal+h+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
                    ['o-proj+down-proj'], ['inter'], ['somemethods-3'],]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['2048'], ['l2'], [f'magstructlocal+w+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
                    ['o-proj+down-proj'], ['full'], ['somemethods-3'],]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
            #         ['inter'], ['somemethods-3'], ['o-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
            #         ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # # ----- opt 1.3b
            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['3'], [f'magunstructglobal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #         ['full'], ['somemethods-3'], ['fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            
            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['3'], [f'magstructglobal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #         ['full'], ['somemethods-3'], ['fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'magstructlocal+w+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #         ['full'], ['somemethods-3'], ['fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #         ['inter'], ['somemethods-3'], ['fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #         ['inter'], ['somemethods-3'], ['fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'w*pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #         ['inter'], ['somemethods-3'], ['fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [9999]],
            #         ['inter'], ['somemethods-3'], ['default', 'fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'w*pqstructlocal+h+2+{x}+-1+max' for x in [9999]],
            #         ['inter'], ['somemethods-3'], ['default', 'fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
            #         ['inter'], ['somemethods-3'], ['fc1', 'fc2', 'fc1+fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]],
            #         ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            
            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
            #         ['inter'], ['somemethods-3'], ['out-proj+fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
            #         ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [9999]],
            #         ['inter'], ['somemethods-3'], ['out-proj+fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [9999]],
            #         ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
            #         ['inter'], ['somemethods-3'], ['out-proj+fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-1.3b'], ['clm'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
            #         ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
        elif 'ic' in data:
            # control_name = [[['CIFAR10', 'CIFAR100'], [ 'resnet18'], ['ic'], ['1'], [f'pqstructlocal:h:2:{x}:1:max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #                  ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['ic'], ['1'], [f'w*pqstructlocal:h:2:{x}:1:max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #                  ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['ic'], ['1'], [f'magstructlocal:h:2:{x}:1:max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #                  ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['ic'], ['1'], [f'magstructlocal:w:2:{x}:1:max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #                  ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['ic'], ['1'], [f'magstructglobal:w:2:{x}:1:max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 9999]],
            #                  ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            

            control_name = [[['CIFAR10', 'CIFAR100'], [ 'resnet18'], ['ic'], ['1'], [f'pqstructlocal+h+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 9999]],
                             ['inter'], ['somemethods-3'], ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['CIFAR10', 'CIFAR100'], [ 'resnet18'], ['ic'], ['1'], [f'magstructlocal+h+2+{x}+1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
                             ['inter'], ['somemethods-3'], ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['ic'], ['1'], [f'w*pqstructlocal:h:2:{x}:1:max' for x in [9999]],
            #                  ['inter'], ['somemethods-3'], ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
        elif 'mc' in data:
            # control_name = [[['piqa', 'storycloze-2016', 'arc-e', 'arc-c', 'hellaswag', 'obqa-main'], ['llama-2-7b'], ['mc'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [9999]],
            #                  ['inter'], ['somemethods-3'], ['down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            control_name = [[['obqa-main'], ['opt-1.3b'], ['mc'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
                    ['inter'], ['somemethods-3'], ['out-proj+fc2']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['obqa-main'], ['opt-1.3b'], ['mc'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
                    ['inter'], ['somemethods-3'], ['out-proj+fc2']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['obqa-main'], ['llama-2-7b'], ['mc'], ['1'], [f'pqstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
                    ['inter'], ['somemethods-3'], ['o-proj+down-proj']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['obqa-main'], ['llama-2-7b'], ['mc'], ['1'], [f'magstructlocal+h+2+{x}+-1+max' for x in [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
                    ['inter'], ['somemethods-3'], ['o-proj+down-proj']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
    elif file == 'test_fix_pruned_model': 
        controls = []
        script_name = [[f'{filename}.py']]
        if 'clm' in data:
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['IFV+4', 'IFV+10', 'IFV+20', 'IFV+40'], [f'pq-nobias+NA+{x}+-100+NA' for x in [0]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['WIFV+4', 'WIFV+10', 'WIFV+20', 'WIFV+40'], [f'pq-nobias+NA+{x}+-100+NA' for x in [0]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['WIFN+4', 'WIFN+10', 'WIFN+20', 'WIFN+40', 'WIFN+100'], [f'pq-nobias+NA+{x}+-100+NA' for x in [0]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)




            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128'], ['WIFN+128'], [f'wandasp+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128', '512', '1024', '2048'], ['WIFV+24', 'WIFV+128'], [f'flap+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128', '512', '1024', '2048'], ['WIFN+24', 'WIFN+128'], [f'wandasp+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128', '512', '1024', '2048'], ['WIFN+24', 'WIFN+128'], [f'pqnobias+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128', '512', '1024', '2048'], ['WIFN+24', 'WIFN+128'], [f'pqnobias-0.5-0.5+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
                    ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128', '512', '1024', '2048'], ['WIFN+24', 'WIFN+128'], [f'pqnobiasglobal-0.5-0.5+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
                    ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128', '512', '1024', '2048'], ['WIFN+24', 'WIFN+128'], [f'pqnobiasnormhead-0.5-0.5+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
                    ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128'], [ 'WIFN+128'], [f'pqnobias-0.5-0.5+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls( script_name, init_seeds, device, resume_mode,control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128'], ['WIFN+128'], [f'pqnobias-0.5-0.5+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128'], ['WIFN+128'], [f'pqnobiasnormhead-0.5-0.5+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128'], ['WIFN+128'], [f'pqnobiasglobal-0.5-0.5+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['WIFN+128'], [f'wanda-sp+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['WIFV+128'], [f'flap+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['IFV+4', 'IFV+10', 'IFV+20', 'IFV+40'], [f'wanda-sp+NA+{x}+-100+NA' for x in [0, 0.2, 0.3, 0.5]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['WIFV+4', 'WIFV+10', 'WIFV+20', 'WIFV+40'], [f'wanda-sp+NA+{x}+-100+NA' for x in [0, 0.2, 0.3, 0.5]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['WIFN+4', 'WIFN+10', 'WIFN+20', 'WIFN+40', 'WIFN+100'], [f'wanda-sp+NA+{x}+-100+NA' for x in [0, 0.2, 0.3, 0.5]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['WIFN+4'], [f'mag-sp+NA+{x}+-100+NA' for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5]],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            pass
    else:
        raise ValueError('Not valid file')

    print('%$%$$controls', controls)

    # k = 0
    # s_for_max = '#!/bin/bash\n'
    # k_for_max = 0
    import os
    res_path = os.path.join(f'/home/aanwar/le000288/{code_folder}/src', res_folder)
    print(res_path)
    makedir_exist_ok(res_folder)

    bash_file_name = './{}.bash'.format(f'msi_{file}_{data[0]}')

    # Check if the file exists
    if os.path.exists(bash_file_name):
        # Delete the file if it exists
        os.remove(bash_file_name)

    task_parallel_num = int(round / num_gpus)
    mem = 15
    if task_parallel_num == 1:
        mem = 15
    elif task_parallel_num == 2:
        mem = 45
    elif task_parallel_num == 3:
        mem = 65
    

    i = 0
    while i < len(controls):
    # for i in range(len(controls)):
        controls[i] = list(controls[i])
        temp = controls[i][3:] 
        filename = ''.join(str(_) for _ in temp)
        filename = pbs_prefix + filename
        # print(f'filename: {filename}')

        sub_controls = []
        j = 0
        is_llama = False
        is_opt = False
        is_gpt = False
        while j < task_parallel_num and i < len(controls):
            controls[i] = list(controls[i])
            sub_controls.append(controls[i])
            print('controls[i]', controls[i])
            if 'llama' in controls[i][4]:
                is_llama = True
            if 'opt' in controls[i][4]:
                is_opt = True
            if 'gpt' in controls[i][4]:
                is_gpt = True
            j += 1
            i += 1
        
        # print('isgpt', is_gpt)
        temp_mem = mem
        if is_llama:
            temp_mem = int(3.5 * mem)
        if is_opt:
            temp_mem = int(1.5 * temp_mem)
        if is_gpt:
            temp_mem = int(1.5 * temp_mem)
        s = '#!/bin/bash -l\n'
        s += '#SBATCH --time=00:30:00\n'
        s += f'#SBATCH --nodes={task_parallel_num}\n'
        s += f'#SBATCH --ntasks={task_parallel_num}\n'
        # s += '#SBATCH --cpus-per-task=2'
        s += '#SBATCH --gres=gpu:a100:1\n'
        s += '#SBATCH --partition=a100-8\n'
        s += f'#SBATCH --mem={temp_mem}gb\n'
        s += '#SBATCH --mail-type=ALL \n'
        s += '#SBATCH --mail-user=le000288@umn.edu\n'
        s += f'#SBATCH -o {res_path}/{filename}_%j.out\n'
        s += f'#SBATCH -e {res_path}/{filename}_%j.err\n'
        s += '\n'
        s += f'cd /home/aanwar/le000288/{code_folder}/src\n'
        s += '\n'
        s += 'export PATH=/home/aanwar/le000288/miniconda3/envs/eri/bin:$PATH\n'
        # if 'max' in controls[i][-1]:
        #     s_for_max = s_for_max + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
        #         '--resume_mode {} --log_interval {} --device {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])

        #     if k_for_max % round == round - 1:
        #         s_for_max = s_for_max[:-2] + '\nwait\n'
        #     k_for_max = k_for_max + 1
        #     continue
        # while i < len(controls):
        # srun --nodes=1 --ntasks=1 
        for item in sub_controls:
            s += '\n'
            s = s + 'srun --nodes=1 --ntasks=1 python {} --device {} --resume_mode {} --init_seed {} --control_name {}\n'.format(*item)
        
        s += 'wait\n'
        # controls[i][0] = 'test_classifier_fl.py'
        # for item in sub_controls:
        #     item[0] = item[0].replace('train', 'test')
        #     print(item, item[0])
        #     s += '\n'
        #     s = s + 'srun --nodes=1 --ntasks=1 python {} --device {} --resume_mode {} --init_seed {} --control_name {}&\n'.format(*item)
        # s += 'wait\n'
        pbs_file_name = './{}.pbs'.format(f'{filename}')
        # Check if the file exists
        if os.path.exists(pbs_file_name):
            # Delete the file if it exists
            os.remove(pbs_file_name)
        run_file = open(pbs_file_name, 'a')
        run_file.write(s)
        run_file.close()

        run_file = open(bash_file_name, 'a')
        command = f'mkdir {res_path}\nsbatch {filename}.pbs --wait\n'
        run_file.write(command)
        run_file.close()
    return


if __name__ == '__main__':
    main()
