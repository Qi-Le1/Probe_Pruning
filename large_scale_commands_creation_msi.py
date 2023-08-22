import argparse
import itertools

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
    filename = '{}_{}'.format(run, file)
    
    if file == 'classifier_fl' or file == 'heuristic_search':
        controls = []
        script_name = [[f'{filename}.py']]
        if 'CIFAR10' in data:
            

            # Fix
            # control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.1'], 
            #                  ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4'], 
            #                  ['6-1', '5-1', '4-1']]]
            # CIFAR10_controls_3 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_3)
            if temp_num_experiments == 3:
                control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                                ['fedprox'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
                CIFAR10_controls_2 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                controls.extend(CIFAR10_controls_2)

                # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['scaffold'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
                # CIFAR10_controls_2 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_2)
                # control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['fedavg'], ['5'], ['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_2 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_2)

                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['fedprox'], ['5'], ['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_3 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_3)

                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['fedensemble'], ['5'], ['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_4 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_4)

                # for scaffold
                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['scaffold'], ['5'], ['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_3 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_3)

                # control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['fedavg', 'fedprox', 'scaffold'], ['5'], ['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_4 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_4)
                
                # control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['fedensemble'], ['5'], ['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_5 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_5)
            # test baseline
            # if temp_num_experiments == 2:
            #     # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
            #     #                 ['fedprox'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
            #     # CIFAR10_controls_3 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            #     # controls.extend(CIFAR10_controls_4)

            #     control_name = [[['CIFAR10', 'CIFAR100'], ['cnn', 'resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
            #                     ['fedavg'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
            #     CIFAR10_controls_3 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            #     controls.extend(CIFAR10_controls_3)

                # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['fedensemble'], ['5'], ['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_4 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_4)

                # for scaffold
                # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['scaffold'], ['5'], ['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_3 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_3)
            if temp_num_experiments == 1:
                # # cifar10_100_dynamicfl_resnet_seed1 
                # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_5 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_5)
                # # cifar10_100_dynamicfl_resnet_seed1 
                # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'],
                #                 ['6-1']]]
                # CIFAR10_controls_6 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_6)

                # femnist_dynamicfl_cnn_seed1 
                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                #                 ['4-1', '5-1', '6-1']]]
                # CIFAR10_controls_5 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_5)
                # # femnist_dynamicfl_cnn_seed1 
                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'],
                #                 ['4-1', '5-1', '6-1']]]
                # CIFAR10_controls_6 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_6)

                # femnist_dynamicfl_res_seed1 
                # control_name = [[['FEMNIST'], ['resnet18'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                #                 ['4-1', '5-1', '6-1']]]
                # CIFAR10_controls_5 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_5)
                # # femnist_dynamicfl_res_seed1 
                # control_name = [[['FEMNIST'], ['resnet18'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'],
                #                 ['4-1', '5-1', '6-1']]]
                # CIFAR10_controls_6 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_6)


                # fesgd
                # msi_bash_both_fedsgd_baseline
                # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18', 'cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_5 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_5)

                # # cifar10_100_dynamicfl_resnet_seed0 
                # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                #                 ['5-1']]]
                # CIFAR10_controls_5 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_5)
                # # # cifar10_100_dynamicfl_resnet_seed0 
                # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                #                 ['5-1']]]
                # CIFAR10_controls_6 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_6)

                # # # cifar10_100_dynamicfl_resnet_seed0 
                # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_7 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_7)
                # # # cifar10_100_dynamicfl_resnet_seed0 
                # control_name = [[['CIFAR10', 'CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                #                 ['6-1']]]
                # CIFAR10_controls_8 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_8)

                # ablation_cifar100
                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['4-1', '3-1'], ['1']]]
                # CIFAR10_controls_7 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_7)

                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['6-5', '6-4', '6-3', '6-2.5', '6-2', '5-4', '5-3', '5-2.5', '5-2', '4-3', '4-2.5', '4-2', '4-1', '3-2.5', '3-2', '3-1']]]
                # CIFAR10_controls_8 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_8)

                # dynamicsgd
                # control_name = [[['CIFAR100', 'CIFAR10'], ['cnn', 'resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicsgd'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_7 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_7)

                # # dynamicsgd
                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicsgd'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_8 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_8)

                # check_communicatoin_cost
                # control_name = [[['CIFAR100', 'CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                #                 ['6-1', '5-1', '4-1'], ['0'], ['1']]]
                # CIFAR10_controls_7 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_7)

                # # check_communicatoin_cost
                # control_name = [[['CIFAR100', 'CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                #                 ['6-1', '5-1', '4-1'], ['0'], ['1']]]
                # CIFAR10_controls_8 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_8)

                # check_communicatoin_cost
                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'],  
                #                 ['6-1', '5-1', '4-1'], ['0'], ['1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # # check_communicatoin_cost
                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                #                 ['6-1', '5-1', '4-1'], ['0'], ['1']]]
                # CIFAR10_controls_10 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_10)

                # # check_communicatoin_cost
                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['6-5', '6-4', '6-3', '6-2.5', '6-2', '6-1', '5-4', '5-3', '5-2.5', '5-2', '5-1', '4-3', '4-2.5', '4-2', '4-1', '3-2.5', '3-2', '3-1'], ['0'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)

                # check all communication cost
                # control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                #                 ['6-1', '5-1',  '4-1', '3-1'], ['0'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)

                # control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'],  
                #                 ['6-1', '5-1',  '4-1', '3-1'], ['0'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)
            
                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                #                 ['6-1', '5-1',  '4-1', '3-1'], ['0'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)

                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'],  
                #                 ['6-1', '5-1',  '4-1', '3-1'], ['0'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)

                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                #                 ['6-1', '5-1',  '4-1', '3-1'], ['0'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)

                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'],  
                #                 ['6-1', '5-1',  '4-1', '3-1'], ['0'], ['1']]]

                # ------

                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)
                # control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['6-1', '5-1', '4-1', '3-1'], ['1'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)

                # control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1', '5-1', '4-1', '3-1'], ['0'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)

                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['6-1', '5-1', '4-1', '3-1'], ['1'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)

                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1', '5-1', '4-1', '3-1'], ['0'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)

                # # check cifar10 femenist communication cost
                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['6-5', '6-4', '6-3', '6-2.5', '6-2', '6-1', '5-4', '5-3', '5-2.5', '5-2', '5-1', '4-3', '4-2.5', '4-2', '4-1', '3-2.5', '3-2', '3-1'], ['0'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)

                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['6-1', '5-1', '4-1', '3-1'], ['1'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)

                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1', '5-1', '4-1', '3-1'], ['0'], ['1']]]
                # CIFAR10_controls_11 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_11)
                # check_dp
                # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn'], ['0.1', '0.3'], ['100'], ['non-iid-l-1', 'non-iid-l-2','non-iid-d-0.1', 'non-iid-d-0.3'], 
                #              ['dynamicfl'], ['5']]]
                # CIFAR10_controls_3 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_3)

                # # check_dp
                # control_name = [[['FEMNIST'], ['cnn'], ['0.1', '0.3'], ['100'], ['non-iid-d-0.01','non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                     ['dynamicfl'], ['5']]]
                # CIFAR10_controls_4 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_4)

                # check_feddyn
                # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn', 'resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['feddyn'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_8 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_8)

                # ablation_cifar10
                # control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['6-1', '5-1'], ['1']]]
                # CIFAR10_controls_7 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_7)

                # # ablation_cifar10
                # control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['4-1', '3-1'], ['1']]]
                # CIFAR10_controls_8 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_8)

                # # ablation_cifar10
                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['6-5', '6-4', '6-3', '6-2.5', '6-2', '5-4', '5-3', '5-2.5', '5-2', '4-3', '4-2.5', '4-2', '4-1', '3-2.5', '3-2', '3-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # # for ablation entire comparison
                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['1-0'], 
                #                 ['5-1', '4-1', '3-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['6-1', '5-1', '4-1', '3-1'], ['1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)
                # femnist ablation
                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['6-1', '5-1', '4-1', '3-1'], ['1']]]
                # CIFAR10_controls_7 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_7)

                # # femnist ablation
                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['5-4', '5-3', '5-2.5', '5-2', '4-3', '4-2.5', '4-2', '4-1', '3-2.5', '3-2', '3-1', '6-5', '6-4', '6-3', '6-2.5', '6-2', ]]]
                # CIFAR10_controls_8 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_8)

                # maoge rest
                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                #                 ['5-1', '4-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # control_name = [[['CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1'], 
                #                 ['dynamicfl'], ['5'],['0.3-0.7', '0.6-0.4'], ['1-0'], 
                #                 ['5-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['0.3-0.7', '0.6-0.4'], ['1-0'], 
                #                 ['5-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2'], 
                #                 ['dynamicfl'], ['5'],['0.9-0.1'], ['1-0'], 
                #                 ['5-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

# key FEMNIST_cnn_0.1_100_non-iid-d-0.3_fedensemble_5_1-0_1-0_6-1
# key FEMNIST_cnn_0.1_100_non-iid-d-0.3_scaffold_5_1-0_1-0_6-1


#                 Missing ./output/result/freq_ablation/0_CIFAR10_cnn_0.1_100_non-iid-l-1_dynamicfl_5_1-0_1-0_5-1.pt
# Missing ./output/result/freq_ablation/0_CIFAR10_cnn_0.1_100_non-iid-l-2_dynamicfl_5_1-0_1-0_4-1.pt
# Missing ./output/result/freq_ablation/0_CIFAR10_cnn_0.1_100_non-iid-l-2_dynamicfl_5_1-0_1-0_3-1.pt
# Missing ./output/result/freq_ablation/0_CIFAR10_cnn_0.1_100_non-iid-d-0.1_dynamicfl_5_1-0_1-0_5-1.pt
# Missing ./output/result/freq_ablation/0_CIFAR10_cnn_0.1_100_non-iid-d-0.3_dynamicfl_5_1-0_1-0_5-1.pt

                # ablation_rest
                # control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['1-0'], 
                #                 ['5-1']]]
                # CIFAR10_controls_9 = make_controls(control_name)
                # controls.extend(CIFAR10_controls_9)

#                 1_CIFAR10_cnn_0.1_100_non-iid-l-1_fedgen_5_1-0_1-0_6-1.pt
# Missing ./output/result/cnn_all/1_CIFAR10_cnn_0.1_100_non-iid-l-2_fedgen_5_1-0_1-0_6-1.pt
# Missing ./output/result/cnn_all/1_CIFAR10_cnn_0.1_100_non-iid-d-0.1_fedgen_5_1-0_1-0_6-1.pt
# Missing ./output/result/cnn_all/1_CIFAR10_cnn_0.1_100_non-iid-d-0.3_fedgen_5_1-0_1-0_6-1.pt
# Missing ./output/result/cnn_all/1_CIFAR100_cnn_0.1_100_non-iid-l-1_fedgen_5_1-0_1-0_6-1.pt
# Missing ./output/result/cnn_all/1_CIFAR100_cnn_0.1_100_non-iid-l-2_fedgen_5_1-0_1-0_6-1.pt
# Missing ./output/result/cnn_all/1_CIFAR100_cnn_0.1_100_non-iid-d-0.1_fedgen_5_1-0_1-0_6-1.pt
# Missing ./output/result/cnn_all/1_CIFAR100_cnn_0.1_100_non-iid-d-0.3_fedgen_5_1-0_1-0_6-1.pt

                # fedgen rest
                # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-1', 'non-iid-l-2'], 
                #                 ['fedgen'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # mimic_fedsgd
                # control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['iid'], 
                #                 ['dynamicsgd'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # mimic_fedsgd_resnet
                # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
                #                 ['dynamicsgd'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)
                # res_seed1_5-1_rest


# Missing ./output/result/resnet18_all/1_CIFAR10_resnet18_0.1_100_non-iid-d-0.1_dynamicfl_5_1-0_0.3-0.7_5-1.pt
# Missing ./output/result/resnet18_all/1_CIFAR10_resnet18_0.1_100_non-iid-d-0.1_dynamicfl_5_1-0_0.6-0.4_5-1.pt
# Missing ./output/result/resnet18_all/1_CIFAR10_resnet18_0.1_100_non-iid-d-0.3_dynamicfl_5_1-0_0.3-0.7_5-1.pt
# Missing ./output/result/resnet18_all/1_CIFAR10_resnet18_0.1_100_non-iid-d-0.3_dynamicfl_5_1-0_0.6-0.4_5-1.pt



# Missing ./output/result/resnet18_all/1_CIFAR100_resnet18_0.1_100_non-iid-l-1_dynamicfl_5_0.3-0.7_1-0_5-1.pt
# Missing ./output/result/resnet18_all/1_CIFAR100_resnet18_0.1_100_non-iid-l-1_dynamicfl_5_0.6-0.4_1-0_5-1.pt


# Missing ./output/result/resnet18_all/1_CIFAR10_resnet18_0.1_100_non-iid-l-1_dynamicfl_5_1-0_0.9-0.1_5-1.pt
# Missing ./output/result/resnet18_all/1_CIFAR10_resnet18_0.1_100_non-iid-l-2_dynamicfl_5_1-0_0.9-0.1_5-1.pt


# Missing ./output/result/resnet18_all/1_CIFAR100_resnet18_0.1_100_non-iid-d-0.1_dynamicfl_5_0.3-0.7_1-0_5-1.pt
# Missing ./output/result/resnet18_all/1_CIFAR100_resnet18_0.1_100_non-iid-d-0.3_dynamicfl_5_0.3-0.7_1-0_5-1.pt

# Missing ./output/result/resnet18_all/1_CIFAR100_resnet18_0.1_100_non-iid-d-0.3_dynamicfl_5_1-0_0.9-0.1_6-1.pt
                # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], [ 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4'], 
                #                 ['5-1']]]
                # CIFAR10_controls_8 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_8)

                # control_name = [[['CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4'], 
                #                 ['5-1']]]
                # CIFAR10_controls_8 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_8)

                # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.9-0.1'], 
                #                 ['5-1']]]
                # CIFAR10_controls_8 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_8)

                # control_name = [[['CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['dynamicfl'], ['5'],['0.3-0.7'], ['1-0'], 
                #                 ['5-1']]]
                # CIFAR10_controls_8 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_8)

                # fednova_rest
                # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.1', 'non-iid-d-0.3'], 
                #                 ['fednova'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # resnet_baseline_rest
                # control_name = [[['CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-d-0.3'], 
                #                 ['feddyn', 'fedgen', 'scaffold'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # control_name = [[['CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-d-0.1', 'non-iid-l-1'], 
                #                 ['fedgen'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-1', 'non-iid-l-2'], 
                #                 ['fedgen'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1'], 
                #                 ['feddyn'], ['5'],['1-0'], ['1-0'], 
                #                 ['6-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)
                
                # freq_ablation_missing_rest
                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-1', 'non-iid-l-2'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['1-0'], 
                #                 ['5-1', '4-1', '3-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-d-0.01'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['1-0'], 
                #                 ['5-1', '4-1', '3-1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1'], 
                #                 ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                #                 ['6-1', '5-1'], ['1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)
                
#                Missing ./output/result/freq_ablation/0_CIFAR100_cnn_0.1_100_non-iid-l-1_dynamicfl_5_1-0_0.3-0.7_6-1_1.pt
# Missing ./output/result/freq_ablation/0_CIFAR100_cnn_0.1_100_non-iid-l-1_dynamicfl_5_1-0_0.3-0.7_5-1_1.pt
# Missing ./output/result/freq_ablation/0_CIFAR100_cnn_0.1_100_non-iid-d-0.3_dynamicfl_5_1-0_0.3-0.7_5-1.pt

                # femnist_cost
                # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                #         ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                #         [ '6-1', '5-1', '4-1'], ['0'], ['1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # 'non-iid-d-0.3', 'non-iid-d-0.1', 
                # control_name = [[['CIFAR10'], ['cnn', 'resnet18'], ['0.1'], ['100'], ['iid'], 
                #                 ['fedavg'], ['5'], ['0', '0.05'], ['100', '250', '500'], ['sparsity', 'PQ'], ['0.05', '0.1']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)



                # control_name = [[['CIFAR10'], ['cnn', 'resnet18'], ['0.1'], ['100'], ['iid'], 
                #                 ['fedavg'], ['5'], ['0'], ['500'], ['sparsity', 'PQ'], ['0']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
                #                 ['fedavg'], ['5'], ['0'], ['1'], ['1', '2'], ['our', 'unstructured', 'channel-wise', 'filter-wise'], ['PQ'],  ['0', '0.001', '0.01', '0.03', '0.06', '0.1', '0.5', '1.0', '999']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)

                control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
                                ['fedavg'], ['5'], ['0'], ['1'], ['2'], ['channel-wise', 'filter-wise'], ['PQ'],  [ '0.01']]]
                CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                controls.extend(CIFAR10_controls_9)

                # control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['iid'], 
                #                 ['fedavg'], ['5'], ['0'], ['10', '100', '1000'], ['1', '2'], ['our'], ['PQ'],  ['0', '0.001', '0.01', '0.03', '0.06', '0.1', '0.5', '1.0', '999'], ['inter']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)
                # control_name = [[['CIFAR10'], ['cnn', 'resnet18'], ['0.1'], ['100'], ['iid'], 
                #                 ['fedavg'], ['5'], ['0'], ['1'], ['PQ'], ['0.03', '0.06']]]
                # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
                # controls.extend(CIFAR10_controls_9)
        if 'CIFAR100' in data:
            '''
            group loss commands
            '''
            control_name = [[['CIFAR100'], ['cnn'], ['0.1', '0.3', '0.5'], ['100'], ['non-iid-l-1', 'non-iid-l-2','non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['nonpre']]]
            CIFAR10_controls_3 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_3)
        if 'FEMNIST' in data:
            '''
            group loss commands
            '''
            control_name = [[['FEMNIST'], ['cnn'], ['0.1', '0.3', '0.5'], ['100'], ['non-iid-l-1', 'non-iid-l-2','non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['nonpre']]]
            CIFAR10_controls_3 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_3)
    else:
        raise ValueError('Not valid file')

    print('%$%$$controls', controls)

    
    # k = 0

    # s_for_max = '#!/bin/bash\n'
    # k_for_max = 0
    import os
    res_path = os.path.join(f'/home/aanwar/le000288/{code_folder}', res_folder)
    print(res_path)
    # os.mkdir(path)

    task_parallel_num = int(round / num_gpus)
    mem = 20
    if task_parallel_num == 1:
        mem = 20
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
        is_resnet = False
        is_femnist = False
        while j < task_parallel_num and i < len(controls):
            controls[i] = list(controls[i])
            sub_controls.append(controls[i])
            print(controls[i])
            if 'resnet' in controls[i][4]:
                is_resnet = True
            if 'FEMNIST' in controls[i][4]:
                is_femnist = True
            j += 1
            i += 1
            
        temp_mem = mem
        if is_resnet:
            temp_mem = int(2 * mem)
        if is_femnist:
            temp_mem = int(3 * temp_mem)
        s = '#!/bin/bash -l\n'
        s += '#SBATCH --time=1:30:00\n'
        s += f'#SBATCH --nodes={task_parallel_num}\n'
        s += f'#SBATCH --ntasks={task_parallel_num}\n'
        # s += '#SBATCH --cpus-per-task=2'
        s += '#SBATCH --gres=gpu:a100:1\n'
        s += '#SBATCH --partition=a100-4\n'
        s += f'#SBATCH --mem={temp_mem}gb\n'
        s += '#SBATCH --mail-type=ALL \n'
        s += '#SBATCH --mail-user=le000288@umn.edu\n'
        s += f'#SBATCH -o {res_path}/{filename}_%j.out\n'
        s += f'#SBATCH -e {res_path}/{filename}_%j.err\n'
        s += '\n'
        s += f'cd /home/aanwar/le000288/{code_folder}\n'
        s += '\n'
        s += 'export PATH=/home/aanwar/le000288/miniconda3/envs/dynamicfl/bin:$PATH\n'
        # if 'max' in controls[i][-1]:
        #     s_for_max = s_for_max + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
        #         '--resume_mode {} --log_interval {} --device {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])

        #     if k_for_max % round == round - 1:
        #         s_for_max = s_for_max[:-2] + '\nwait\n'
        #     k_for_max = k_for_max + 1
        #     continue
        # while i < len(controls):
        
        for item in sub_controls:
            s += '\n'
            s = s + 'srun --nodes=1 --ntasks=1 python {} --device {} --resume_mode {} --init_seed {} --control_name {}&\n'.format(*item)
        
        s += 'wait\n'
        # controls[i][0] = 'test_classifier_fl.py'

        # TODO
        # for item in sub_controls:
        #     item[0] = item[0].replace('train', 'test')
        #     print(item, item[0])
        #     s += '\n'
        #     s = s + 'srun --nodes=1 --ntasks=1 python {} --device {} --resume_mode {} --init_seed {} --control_name {}&\n'.format(*item)
        # s += 'wait\n'


        run_file = open('./{}.pbs'.format(f'{filename}'), 'a')
        run_file.write(s)
        run_file.close()

        run_file = open('./{}.bash'.format(f'most_acc_weight'), 'a')
        command = f'mkdir {res_path}\nsbatch {filename}.pbs --wait\n'
        run_file.write(command)
        run_file.close()

        # if 'non-iid-l' in filename:
        #     s = '#!/bin/bash -l\n'
        #     s += '#SBATCH --time=23:59:59\n'
        #     s += '#SBATCH --nodes=1\n'
        #     s += '#SBATCH --ntasks-per-node=1\n'
        #     s += '#SBATCH --gres=gpu:a100:1\n'
        #     s += '#SBATCH --partition=a100-4\n'
        #     s += '#SBATCH --mem=100gb\n'
        #     s += '#SBATCH --mail-type=ALL \n'
        #     s += '#SBATCH --mail-user=le000288@umn.edu\n'
        #     s += '#SBATCH -o %j.out\n'
        #     s += '#SBATCH -e %j.err\n'
        #     s += '\n'
        #     s += 'cd /home/aanwar/le000288/src\n'
        #     s += '\n'
        #     s += 'export PATH=/home/aanwar/le000288/anaconda3/envs/dynamicfl/bin:$PATH\n'
        #     # if 'max' in controls[i][-1]:
        #     #     s_for_max = s_for_max + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
        #     #         '--resume_mode {} --log_interval {} --device {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])

        #     #     if k_for_max % round == round - 1:
        #     #         s_for_max = s_for_max[:-2] + '\nwait\n'
        #     #     k_for_max = k_for_max + 1
        #     #     continue
        #     s += '\n'
        #     s = s + 'python {} --device {} --control_name {}\n'.format(*controls[i])
        #     s += 'wait\n'
        #     controls[i][0] = 'test_classifier_fl.py'
        #     s = s + 'python {} --device {} --control_name {}\n'.format(*controls[i])
            
        #     run_file = open('./{}.pbs'.format(f'{filename}'), 'a')
        #     run_file.write(s)
        #     run_file.close()

        #     run_file = open('./{}.bash'.format(f'msi_bash_non_iid_l'), 'a')
        #     command = f'sbatch {filename}.pbs\n'
        #     run_file.write(command)
        #     run_file.close()
        # else:
        #     s = '#!/bin/bash -l\n'
        #     s += '#SBATCH --time=23:59:59\n'
        #     s += '#SBATCH --nodes=1\n'
        #     s += '#SBATCH --ntasks-per-node=1\n'
        #     s += '#SBATCH --gres=gpu:a100:1\n'
        #     s += '#SBATCH --partition=a100-8\n'
        #     s += '#SBATCH --mem=100gb\n'
        #     s += '#SBATCH --mail-type=ALL \n'
        #     s += '#SBATCH --mail-user=le000288@umn.edu\n'
        #     s += '#SBATCH -o %j.out\n'
        #     s += '#SBATCH -e %j.err\n'
        #     s += '\n'
        #     s += 'cd /home/aanwar/le000288/src\n'
        #     s += '\n'
        #     s += 'export PATH=/home/aanwar/le000288/anaconda3/envs/dynamicfl/bin:$PATH\n'
        #     # if 'max' in controls[i][-1]:
        #     #     s_for_max = s_for_max + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
        #     #         '--resume_mode {} --log_interval {} --device {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])

        #     #     if k_for_max % round == round - 1:
        #     #         s_for_max = s_for_max[:-2] + '\nwait\n'
        #     #     k_for_max = k_for_max + 1
        #     #     continue
            
        #     s += '\n'
        #     s = s + 'python {} --init_seed {} --world_size {} --num_experiments {} ' \
        #             '--resume_mode {} --log_interval {} --device {} --control_name {}\n'.format(*controls[i])
        #     s += 'wait\n'
        #     controls[i][0] = 'test_classifier_fl.py'
        #     s = s + 'python {} --init_seed {} --world_size {} --num_experiments {} ' \
        #             '--resume_mode {} --log_interval {} --device {} --control_name {}\n'.format(*controls[i])
            
        #     run_file = open('./{}.pbs'.format(f'{filename}'), 'a')
        #     run_file.write(s)
        #     run_file.close()

        #     run_file = open('./{}.bash'.format(f'msi_bash_other'), 'a')
        #     command = f'sbatch {filename}.pbs\n'
        #     run_file.write(command)
        #     run_file.close()

        # if k % round == round - 1:
        #     s = s[:-2] + '\nwait\n'
        # k = k + 1

    # if s[-5:-1] != 'wait':
    #     s = s + 'wait\n'
    # if s_for_max != '#!/bin/bash\n' and s_for_max[-5:-1] != 'wait':
    #     s_for_max = s_for_max + 'wait\n'
    
    # print('@@@@@@@@@@', s)
    

    # run_file = open('./{}.sh'.format('large_scale_one_user_per_node_commands'), 'a')
    # run_file.write(s_for_max)
    # run_file.close()

    # server_total = 5
    # if run == 'train':
    #     filename = 'train_server_commands'
    #     for i in range(1, server_total):
    #         run_file = open('./{}.sh'.format(f'large_scale_train_server_{i}'), 'a')
    #         run_file.write('#!/bin/bash\n')
    #         run_file.close()

    #         # run_file.write(s)
    #         # run_file.close()
        
    #     run_file = open('./{}.txt'.format(f'large_scale_{filename}_temp'), 'a')
    #     run_file.write(s)
    #     run_file.close()
    # elif run == 'test':
    #     filename = 'test_server_commands'
    #     for i in range(1, server_total):
    #         run_file = open('./{}.sh'.format(f'large_scale_test_server_{i}'), 'a')
    #         run_file.write('#!/bin/bash\n')
    #         run_file.close()

    #         # run_file.write(s)
    #         # run_file.close()

    #     run_file = open('./{}.txt'.format(f'large_scale_{filename}_temp'), 'a')
    #     run_file.write(s)
    #     run_file.close()
    

    # print(f'sss: {s}')
    # run_file = open('./{}.sh'.format(f'large_scale_{filename}'), 'a')
    # run_file.write(s)
    # run_file.close()

    # run_file = open('./{}.txt'.format(f'large_scale_{run}'), 'a')
    # run_file.write(s_for_max)
    # run_file.close()

    

        

    # new_s = s.replace('CUDA_VISIBLE_DEVICES="0" ', '!')
    # new_s = new_s.replace('CUDA_VISIBLE_DEVICES="1" ', '!')
    # new_s = new_s.replace('CUDA_VISIBLE_DEVICES="2" ', '!')
    # new_s = new_s.replace('CUDA_VISIBLE_DEVICES="3" ', '!')
    # print('????', new_s)
    # run_file = open('./{}.sh'.format(f'pre_run_large_scale_{filename}'), 'a')
    # run_file.write(new_s)
    # run_file.close()

    # new_s_for_max = s_for_max.replace('CUDA_VISIBLE_DEVICES="0" ', '!')
    # new_s_for_max = new_s_for_max.replace('CUDA_VISIBLE_DEVICES="1" ', '!')
    # new_s_for_max = new_s_for_max.replace('CUDA_VISIBLE_DEVICES="2" ', '!')
    # new_s_for_max = new_s_for_max.replace('CUDA_VISIBLE_DEVICES="3" ', '!')
    # run_file = open('./{}.txt'.format(f'pre_run_large_scale_{run}'), 'a')
    # run_file.write(new_s_for_max)
    # run_file.close()

    # for i in range(4, 4):
    #     run_file = open('./{}.sh'.format(f'pre_run_large_scale_train_server_{i}'), 'a')
    #     run_file.write('#!/bin/bash\n')
    #     run_file.close()

    #     run_file = open('./{}.sh'.format(f'pre_run_large_scale_test_server_{i}'), 'a')
    #     run_file.write('#!/bin/bash\n')
    #     run_file.close()

    return


if __name__ == '__main__':
    main()
