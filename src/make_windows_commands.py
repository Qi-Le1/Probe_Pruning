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
    print('file', file)
    data = args['data'].split('_')
    data = [x.replace('\r', '') for x in data]
    print('data', data)
    
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    log_interval = [[log_interval]]
    device = [[device]]
    filename = '{}'.format(file)

    def make_controls(script_name=None, init_seeds=None, device=None, resume_mode=None, control_name=None):
        # global data
        control_names = []
        if 'missing' in data:
            control_names = [
                'wikitext-2v1_llama-2-7b_clm_1_1024_WIFV+128_pq-0.9-0.9-global-std+NA+0.2+-100+NA_down-proj',
                'wikitext-2v1_llama-2-7b_clm_1_1024_WIFN+128_pq-0.9-0.9+NA+0.3+-100+NA_down-proj',
                'wikitext-2v1_llama-2-7b_clm_1_1024_WIFN+128_pq-0.9-0.9+NA+0.4+-100+NA_down-proj',
                'wikitext-2v1_llama-2-7b_clm_1_1024_WIFN+128_pq-0.9-0.9+NA+0.5+-100+NA_down-proj',
                'wikitext-2v1_llama-2-7b_clm_1_1024_WIFN+128_pq-0.9-0.9+NA+0.6+-100+NA_down-proj',
                'wikitext-2v1_llama-2-7b_clm_1_2048_WIFN+128_pq-0.9-0.9+NA+0+-100+NA_down-proj',
                'wikitext-2v1_llama-2-7b_clm_1_2048_WIFN+128_pq-0.9-0.9+NA+0.1+-100+NA_down-proj',
                'wikitext-2v1_llama-2-7b_clm_1_2048_WIFN+128_pq-0.9-0.9+NA+0.2+-100+NA_down-proj',
                'wikitext-2v1_llama-2-7b_clm_1_2048_WIFN+128_pq-0.9-0.9+NA+0.3+-100+NA_down-proj',
                'wikitext-2v1_llama-2-7b_clm_1_2048_WIFN+128_pq-0.9-0.9+NA+0.4+-100+NA_down-proj',
                'wikitext-2v1_llama-2-7b_clm_1_2048_WIFN+128_pq-0.9-0.9+NA+0.5+-100+NA_down-proj',
                'wikitext-2v1_llama-2-7b_clm_1_2048_WIFN+128_pq-0.9-0.9+NA+0.6+-100+NA_down-proj',
        ]
        else:
            for i in range(len(control_name)):
                control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
        control_names = [control_names]
        controls = script_name + device + resume_mode + init_seeds + control_names 
        controls = list(itertools.product(*controls))
        # print('---controls', controls)
        return controls
    
    print('file==test_model', file == 'test_model')
    if file == 'test_model':
        controls = []
        script_name = [[f'{filename}.py']]
        if 'clm' in data:
            print('here')
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probe-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probe-fill-each', 'mag-probe-fill-each-delseq'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probembsz-fill-each', 'mag-probembsz-fill-each-delseq', 'mag-probe-fill-each-delseq'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probembsz-fill-each', 'mag-probe-fill-each', 'mag-probembsz-fill-each-delseq'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probeoptim-fill-each'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probe-None-None', 'mag-probembsz-None-None', 'mag-probembszmseq-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probembsz-fill-each-onlyvo', 'mag-probembsz-fill-each-delseq-onlyvo'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probembszwandasp-fill-each', 'mag-probembszwandasp-fill-each-delseq'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0'], ['pq-probe-None-None-low', 'pq-probewandasp-None-None-low'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0'], ['pq-probeoptim-None-None-low', 'pq-probewandasp-None-None-low'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0'], ['pq-probembsz-fill-fill-low', 'pq-probembsz-fill-fill-delseq-low'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0'], ['pq-probembszwandasp-fill-fill-low', 'pq-probembszwandasp-fill-fill-delseq-low'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probe-None-None-restore', 'mag-probembsz-None-None-restore', 'mag-probembszmseq-None-None-restore'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probembsz-fill-each-restore'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probekeepseq-None-None', 'mag-probekeepseqmbsz-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probembsz-fill-each-onlyvo', 'mag-probembsz-fill-each-onlyqk'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['300'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probe-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0', '0.1', '0.2'], ['pq-probeoptim-fill-fill-low'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0', '0.1', '0.2'], ['pq-probembsz-fill-fill-low'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0', '0.1', '0.2'], ['pq-probembszkeepseq-fill-fill-low'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name) 
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0', '0.1', '0.2'], ['pq-probeoptimwandasp-fill-fill-low'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95'], ['mag-probeoptim-fill-each-onlyvo'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95'], ['mag-probeoptim-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128', '512'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'], ['mag-probembsz-fill-each-onlyvo', 'mag-probenmlmbsz-fill-each-onlyvo'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128', '512'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'], ['mag-probembsz-fill-each-onlyqk', 'mag-probenmlmbsz-fill-each-onlyqk'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128', '512'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'], ['mag-probembsz-fill-each', 'mag-probenmlmbsz-fill-each'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128', '512'], ['0.3', '0.4', '0.5', '0.6'], ['mag-probembszkeepseq-None-None'],
            #         ['gate-proj+up-proj+down-proj', 'gate-proj+down-proj', 'up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128', '512'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], ['mag-probe-None-None'],
            #         [ 'gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['200'], ['128', '512'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], ['mag-probe-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['1024', '2048'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probesvd0.01-None-None', 'mag-probesvd0.05-None-None', 'mag-probesvd0.07-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '60'], ['1024'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probesvd0.01-None-None', 'mag-probesvd0.05-None-None', 'mag-probesvd0.07-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['2048'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probesvd0.01-None-None', 'mag-probesvd0.05-None-None', 'mag-probesvd0.07-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-13b'], ['clm'], ['10'], ['128'], ['0.1'], ['mag-probesvd0.01-probe-None-None'],
            #         ['fc1+fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128', '1024'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], ['mag-runningmean-probe-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probesvd0.01-None-None', 'mag-probesvd0.05-None-None', 'mag-probesvd0.07-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '100'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probesvd1-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-13b'], ['clm'], ['10'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], ['mag-probeoptim-fill-each'],
            #         ['fc1+fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'], ['mag-probe-None-None', 'mag-probembszkeepseq-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128', '512', '1024'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probe-None-None', 'mag-probenmlmbszkeepseq-None-None', 'mag-probestdmbszkeepseq-None-None', 'mag-probelogmbszkeepseq-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-13b'], ['clm'], ['1', '2', '4', '6', '8', '10'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'], ['mag-probe-None-None', 'mag-probembszkeepseq-None-None'],
            #         ['fc1+fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-13b'], ['clm'], ['10'], ['128', '512'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'], ['mag-probewandasp-None-None'],
            #         ['fc1+fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128', '512'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'], ['mag-probembsz-fill-each-onlyqk', 'mag-probembszcompressseq-fill-each-onlyqk', 'mag-probembsz-each-each-onlyqk', 'mag-probembszcompressseq-each-each-onlyqk'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['1024'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probewandasp-None-None', 'mag-probewandaspmbszkeepseq-None-None', 'mag-probewandaspkeepseq-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # calib (calibdataset-all), fixprune, runningmean, prune_metric放第三个, probe默认meanbsz, fillpbmetric
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], 
            #                  ['mag-nmlprobe0.9calib-probe-None-None+wikivalid-all', 'mag-nmlprobe0.9calib-probe-None-None+wikitest-all', 'mag-runningmean-probe-None-None', 'mag-calib-probe-None-None+wikivalid-all', 'mag-calib-probe-None-None+wikitest-all'
            #                   'mag-nmlprobe0.9runningmean-probe-None-None', 'mag-nmlprobe0.9runningmeanfillpbmetric-probe-None-None', 'mag-nmlprobe0.9calibrunningmean-probe-None-None+wikitest-all', 'mag-probe-probe-None-None'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], 
            #                  ['mag-probe0.9runningmean-probe-None-None', 'mag-probe0.9calib-probe-None-None+wikivalid-all', 'mag-calib-probe-None-None+wikivalid-all', 'mag-calib-probe-None-None+wikitest-all', 'mag-runningmean-probe-None-None'
            #                  'mag-probe0.9calib-probe-None-None+wikitest-all', 'mag-probe-probe-None-None'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.5', '0.6'], 
            #                  ['mag-nmlprobe0.9calib-probe-None-None+wikitest-all', 'mag-nmlprobe0.5calib-probe-None-None+wikitest-all', 'mag-calib-probe-None-None+wikitest-all', 'mag-probe0.9calib-probe-None-None+wikitest-all'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.5', '0.6'], 
            #                  ['mag-probefullinf-probe-None-None'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.6'], 
            #                  ['mag-pcabszseqprobe-probe-None-None', 'mag-probe-probe-None-None'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.7', '0.8'], 
            #                  ['mag-pcabszseqprobe0.9calib-probe-None-None+wikitest-all', 'mag-calib-probe-None-None+wikitest-all', 'mag-probefullinf-probe-None-None', 'mag-pcabszseqprobe0.8calib-probe-None-None+wikitest-all'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.7', '0.8'], 
            #                  [ 'mag-fullinfprobe0.1calib-probe-None-None+wikitest-all', 'mag-fullinfprobe0.1calib-probe-None-None+wikitest-all'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.7'], 
            #                  ['mag-probefullinf-probe-None-None+wikitest-all', 'mag-calib-probe-None-None+wikitest-all', 'mag-savemetricseqnmlprobe0.9calib-probe-None-None+wikitest-all',\
            #                    'mag-similarityprobe-probe-None-None+wikitest-all', 'mag-similarityprobe0.9calib-probe-None-None+wikitest-all', 'mag-normbszprobe0.9calib-probe-None-None+wikitest-all'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['50'], ['128'], ['0.7'], 
            #                  ['mag-probefullinf-probe-None-None+wikitest-all', 'mag-calib-probe-None-None+wikitest-all', 'mag-savemetricseqnmlprobe0.9calib-probe-None-None+wikitest-all',\
            #                    'mag-similarityprobe-probe-None-None+wikitest-all', 'mag-similarityprobe0.9calib-probe-None-None+wikitest-all', 'mag-normbszprobe0.9calib-probe-None-None+wikitest-all',
            #                    'mag-similarityprobe0.9calib-probe-None-None+wikitest-all', 'pqlow-similarityprobe0.9calib-probe-None-None+wikitest-all', 'pqlow-savemetricseqnmlprobe0.9calib-probe-None-None+wikitest-all',
            #                    'pq-savemetricseqnmlprobe0.9calib-probe-None-None+wikitest-all'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # 'mag-similarityprobe0.9calibskip5-probe-None-None+wikitest-all','mag-similarityprobe0.9calib-probe-None-None+wikitest-all',
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.6'], 
            #                  ['mag-similarityprobe0.9calib-probe-None-None+wikitest-all', 'mag-nmlprobe0.9calib-probe-None-None+wikitest-all', 'mag-calib-probe-None-None+wikitest-all', 'mag-savemetricseqnmlprobe0.9calib-probe-None-None+wikitest-all', 'mag-savemetricseqnmlprobe0.8calib-probe-None-None+wikitest-all', 'mag-savemetricseqnmlprobe0.7calib-probe-None-None+wikitest-all'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.6'], 
            #                  ['mag-globalratiostdsavemetricseqnmlprobe0.9calib-probe-None-None+c4-2000', 'mag-globalratiostdsavemetricseqnmlprobe0.9calib-probe-None-None+wikitest-all', 'mag-globalratiostdsavemetricseqnmlprobe0.9calib-probe-None-None+wikivalid-all'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.6'], 
            #                  ['mag-globalratiostd0.9calib-probe-None-None+c4-2000', 'mag-globalratiostdnmlprobe0.9calib-probe-None-None+c4-2000', 'mag-globalratiostdsavemetricseqv1nmlprobe0.9calib-probe-None-None+c4-2000',
            #                   'mag-globalratiostdnmlprobe0.9calibrunningmean-probe-None-None+c4-2000', 'mag-globalratiostdnmlprobe0.9calibrunningmeanfillpbmetric-probe-None-None+c4-2000',
            #                   'mag-globalratiostd0.9calib-probe-None-None+wikitest-all', 'mag-globalratiostdnmlprobe0.9calib-probe-None-None+wikitest-all', 'mag-globalratiostdsavemetricseqnmlprobe0.9calib-probe-None-None+wikitest-all',
            #                   'mag-globalratiostd0.9calib-probe-None-None+wikivalid-all', 'mag-globalratiostdnmlprobe0.9calib-probe-None-None+wikivalid-all', 'mag-globalratiostdsavemetricseqnmlprobe0.9calib-probe-None-None+wikivalid-all',],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.6'], 
            #                  [    'mag-globalratiostdsavemetricseqv2rationmlprobe0.9calibcoeffrunningmean-probe-None-None+c4-2000', 'mag-globalratiostdsavemetricseqv2rationmlprobe0.9calibrunningmean-probe-None-None+c4-2000'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.6'], 
            #                  ['pq-nmlprobe0.9calib-probe-None-None+c4-2000'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            
            
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.7', '0.8'], 
            #                  ['mag-probesquare-probe-None-None', 'mag-probe-probe-None-None'],
            #                 ['up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.6', '0.7', '0.8', '0.9'],
#             control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], [ '0.7'], 
#                              [
#                             #      'mag-calib-probe-None-None+c4-2000',
#                             #   'mag-calibrunningmean-probe-None-None+c4-2000',
#                             'mag-calibema0.99-probe-None-None+c4-2000',
#                             'mag-savemetricseqcalibema0.99-probe-None-None+c4-2000',
#                             'mag-globalratiostdcalibema0.99-probe-None-None+c4-2000',
#                             # 'mag-nmlprobe-probe-None-None+c4-2000',
#                             # 'mag-similarityprobedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

#                             # async probe with full inf fill and no clib (totally last round)
#                             # 'mag-nmlhalfsquareasync0.0multiproble10probesavemetricseq-probe-None-None+c4-2000',
#                             # 'mag-nmlhalfsquareasync0.0multiproble10probesavemetricseqema0.99calib-probe-None-None+c4-2000',

#                             # async probe with full inf fill
#                             # 'mag-nmlhalfsquareasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlhalfsquareasync0.3probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

#                             # async probe with different momentum
#                             # 'mag-nmlsquareasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             'mag-nmlsquareasync0.3probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlsquareasync0.5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlsquareasync0.8probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

#                             # # async multi probe
#                             # 'mag-nmlsquareasync0.0multiproble2probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlsquareasync0.0multiproble5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlsquareasync0.0multiproble10probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlsquareasync0.0multiproble10probedynaratiosavemetricseq-probe-None-None+c4-2000',

#                             # # sync, 对比一下不加calib
#                             # 'mag-nmlmultiprobe5probesavemetricseq-probe-None-None+c4-2000',
#                             # 'mag-nmlmultiprobe2probesavemetricseq-probe-None-None+c4-2000',
   
#                             # # sync, 验证当前的epoch multiprobe
#                             'mag-fullinfprobe-probe-None-None+c4-2000',
#                             'mag-nmlmultiprobe10globalratiostdprobedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlmultiprobe5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlmultiprobe2probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             'mag-nmlmultiprobe1probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             'mag-nmlmultiprobe1probedynaratiosavemetricseqrunningmeancalib-probe-None-None+c4-2000',


#                             # 'mag-nmlprobedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

#                             # 'mag-nmlprobedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlsquareasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlsquareasync0.3probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlsquareasync0.5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlsquareasync0.8probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlasync0.3probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlasync0.5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlasync0.8probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
# #                             'mag-nmldynaprobedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
# # #
# #                             'mag-probefullinf-probe-None-None+c4-2000',
                            
#                             #  'mag-nmlprobedynaratiosavemetricseqfillpbmetriccombineema0.99calib-probe-None-None+c4-2000',
#                             #   'mag-nmlprobedynaratiosavemetricseqfillpbmetricoriginalema0.99calib-probe-None-None+c4-2000'
#                             # # 'mag-nmlpr
#                             # obedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlprobedynaratiosavemetricseqrunningmeancalib-probe-None-None+c4-2000',

#                             # 'mag-globalratiostdema0.99calib-probe-None-None+c4-2000',
#                             # # 'mag-nmlprobedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-globalratiostdnmlprobedynaratiosavemetricseqrunningmeancalib-probe-None-None+c4-2000',
#                             # 'mag-nmlprobedynaratiosavemetricseqfillpbmetricoriginalema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlprobedynaratiosavemetricseqcompressfillpbmetriccombineema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlprobedynaratiosavemetricseqfillpbmetricoriginalema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlprobedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
#                             # 'mag-nmlprobedynaratiosavemetricseqcompressfillpbmetricoriginalema0.99calib-probe-None-None+c4-2000',
#                             ],
#                             #   'mag-globalratiostdcalib-probe-None-None+c4-2000',
#                             #   'mag-nmlprobe0.9calib-probe-None-None+c4-2000',
#                             #   'mag-globalratiostdnmlprobe0.9calib-probe-None-None+c4-2000',
#                             #   'mag-globalratiostdsavemetricseqv2rationmlprobe0.9calib-probe-None-None+c4-2000',
#                             #   'mag-globalratiostdsavemetricseqv2rationmlprobe0.9calibrunningmean-probe-None-None+c4-2000'],
#                             ['gate-proj+up-proj+down-proj']]]
#             CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
#             controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.0', '0.3', '0.6', '0.8'], 
                             ['probe'], ['calib-ema'], ['sync', 'asyncinter'], ['c4-15'], ['None'],
                            #  [
                            #     #  'mag-calib-probe-None-None+c4-2000',
                            # #   'mag-calibrunningmean-probe-None-None+c4-2000',
                            # # 'mag-calibnoqkema0.99-probe-None-None-fill+c4-2000',
                            # 'mag-calibnoqkema0.99-probe-None-None-each+c4-2000',

                            # 'mag-calibema0.99-probe-None-None-fill+c4-2',
                            # 'mag-calibema0.99-probe-None-None-fill+c4-2000',
                            # 'mag-calib-probe-None-None-fill+c4-2000',
                            # # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe1each+c4-2000',

                            # # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe1fill+c4-2000',
                            # # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe5each+c4-2000',
                            # # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe1None-probe1None-probe1each+c4-2000',

                            # # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe10each+c4-2000',

                            # # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe1None-probe1None-probe10each+c4-2000',

                            # # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10whole-probe10whole-probe1whole+c4-2000',
                            # # 'mag-calibema0.99-probe-None-None-whole+c4-2000',

                            # # 'mag-calibema0.99noqk-probe-None-None-fill+c4-2000',
                            # # 'mag-calibnoqk-probe-None-None-fill+c4-2000',
                            # # 'mag-calibnoqk-probe-None-None-each+c4-2000',

                            # # 'mag-globalratiostdcalibema0.99noqk-probe-None-None-fill+c4-2000',
                            # # 'mag-calibema0.99noqk-probe-None-None-each+c4-2000',
                            # # 'mag-calibema0.99noqk-probe-None-None-whole+c4-2000',
                            # # 'mag-savemetricseqcalibema0.99-probe-None-None+c4-2000',
                            # # 'mag-globalratiostdcalibema0.99-probe-None-None+c4-2000',
                            
                            # ],
        
                            ['gate-proj+up-proj+down-proj']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.0', '0.3', '0.6', '0.8'], 
            #                  [
            #                     #  'mag-calib-probe-None-None+c4-2000',
            #                 #   'mag-calibrunningmean-probe-None-None+c4-2000',
            #                 # 'mag-calibnoqkema0.99-probe-None-None-fill+c4-2000',
            #                 'mag-calibnoqkema0.99-probe-None-None-each+c4-2000',

            #                 'mag-calibema0.99-probe-None-None-fill+c4-2',
            #                 'mag-calibema0.99-probe-None-None-fill+c4-2000',
            #                 'mag-calib-probe-None-None-fill+c4-2000',
            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe1each+c4-2000',

            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe1fill+c4-2000',
            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe5each+c4-2000',
            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe1None-probe1None-probe1each+c4-2000',

            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe10each+c4-2000',

            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe1None-probe1None-probe10each+c4-2000',

            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10whole-probe10whole-probe1whole+c4-2000',
            #                 # 'mag-calibema0.99-probe-None-None-whole+c4-2000',

            #                 # 'mag-calibema0.99noqk-probe-None-None-fill+c4-2000',
            #                 # 'mag-calibnoqk-probe-None-None-fill+c4-2000',
            #                 # 'mag-calibnoqk-probe-None-None-each+c4-2000',

            #                 # 'mag-globalratiostdcalibema0.99noqk-probe-None-None-fill+c4-2000',
            #                 # 'mag-calibema0.99noqk-probe-None-None-each+c4-2000',
            #                 # 'mag-calibema0.99noqk-probe-None-None-whole+c4-2000',
            #                 # 'mag-savemetricseqcalibema0.99-probe-None-None+c4-2000',
            #                 # 'mag-globalratiostdcalibema0.99-probe-None-None+c4-2000',
                            
            #                 ],
        
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)



            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.3', '0.6', '0.8'], 
            #                  [
            #                     #  'mag-calib-probe-None-None+c4-2000',
            #                 #   'mag-calibrunningmean-probe-None-None+c4-2000',
            #                 # 'mag-calibnoqkema0.99-probe-None-None-fill+c4-2000',
            #                 'mag-calibnoqkema0.99-probe-None-None-each+c4-2000',

            #                 'mag-calibema0.99-probe-None-None-fill+c4-2000',
            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe1each+c4-2000',

            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe1fill+c4-2000',
            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe5each+c4-2000',
            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe1None-probe1None-probe1each+c4-2000',

            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe10each+c4-2000',

            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe1None-probe1None-probe10each+c4-2000',

            #                 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10whole-probe10whole-probe1whole+c4-2000',
            #                 # 'mag-calibema0.99-probe-None-None-whole+c4-2000',

            #                 # 'mag-calibema0.99noqk-probe-None-None-fill+c4-2000',
            #                 # 'mag-calibnoqk-probe-None-None-fill+c4-2000',
            #                 # 'mag-calibnoqk-probe-None-None-each+c4-2000',

            #                 # 'mag-globalratiostdcalibema0.99noqk-probe-None-None-fill+c4-2000',
            #                 # 'mag-calibema0.99noqk-probe-None-None-each+c4-2000',
            #                 # 'mag-calibema0.99noqk-probe-None-None-whole+c4-2000',
            #                 # 'mag-savemetricseqcalibema0.99-probe-None-None+c4-2000',
            #                 # 'mag-globalratiostdcalibema0.99-probe-None-None+c4-2000',
                            
            #                 ],
        
            #                 ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-6.7b'], ['clm'], ['10'], ['128'], ['0.0', '0.2', '0.3', '0.7', '0.8'], 
            #                  [
            #                     #  'mag-calib-probe-None-None+c4-2000',
            #                 #   'mag-calibrunningmean-probe-None-None+c4-2000',
            #                 # 'mag-calibnoqkema0.99-probe-None-None-fill+c4-2000',
            #                 'mag-calibema0.99-probe-None-None-None+c4-2000',
            #                 # 'mag-similarityprobedynaratiosavemetricseqema0.99calib-probe-None-None-None+c4-2000',

            #                 # 'mag-nmlsquareasync0.0multiprobe5probedynaratiosavemetricseqema0.99calib-probe-None-None-None+c4-2000',
            #                 # 'mag-maxmultiprobe1probedynaratiosavemetricseqema0.99calib-probe-None-None-None+c4-2000',

            #                 # 'mag-maxmultiprobe1probemaxsavemetricseqema0.99calib-probe-None-None-None+c4-2000',

            #                 'mag-nmlmultiprobe1probedynaratiosavemetricseqema0.99calib-probe-None-None-None+c4-2000',
            #                 # 'mag-calibema0.99noqk-probe-None-None-fill+c4-2000',
            #                 # 'mag-calibnoqk-probe-None-None-fill+c4-2000',
            #                 # 'mag-calibnoqk-probe-None-None-each+c4-2000',

            #                 # 'mag-globalratiostdcalibema0.99noqk-probe-None-None-fill+c4-2000',
            #                 # 'mag-calibema0.99noqk-probe-None-None-each+c4-2000',
            #                 # 'mag-calibema0.99noqk-probe-None-None-whole+c4-2000',
            #                 # 'mag-savemetricseqcalibema0.99-probe-None-None+c4-2000',
            #                 # 'mag-globalratiostdcalibema0.99-probe-None-None+c4-2000',
                            
            #                 ],
        
            #                 ['fc1+fc2']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.6'], 
            #                  [
            #                 #      'mag-calib-probe-None-None+c4-2000',
            #                 #   'mag-calibrunningmean-probe-None-None+c4-2000',
            #                 'mag-calibema0.99-probe-None-None-fill+c4-2000',
            #                 'mag-calibema0.99-probe-None-None-each+c4-2000',
            #                 'mag-calibema0.99noqk-probe-None-None-fill+c4-2000',
            #                 'mag-calibema0.99noqk-probe-None-None-each+c4-2000',
            #                 'mag-calibema0.99noqk-probe-None-None-whole+c4-2000',
            #                 # 'mag-savemetricseqcalibema0.99-probe-None-None+c4-2000',
            #                 # 'mag-globalratiostdcalibema0.99-probe-None-None+c4-2000',
                            
            #                 ],
        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], 
            #                  [
            #                 #      'mag-calib-probe-None-None+c4-2000',
            #                 #   'mag-calibrunningmean-probe-None-None+c4-2000',
            #                 # 'mag-calibema0.99-probe-None-None+c4-2000',
            #                 # 'mag-nmlprobe-probe-None-None+c4-2000',
            #                 # 'mag-similarityprobedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

            #                 # async probe with full inf fill and no clib (totally last round)
            #                 'mag-nmlhalfsquareasync0.0multiproble10probesavemetricseq-probe-None-None+c4-2000',
            #                 'mag-nmlhalfsquareasync0.0multiproble10probesavemetricseqema0.99calib-probe-None-None+c4-2000',

            #                 # async probe with full inf fill
            #                 'mag-nmlhalfsquareasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlhalfsquareasync0.3probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

            #                 # async probe with different momentum
            #                 'mag-nmlsquareasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.3probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.8probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

            #                 # async multi probe
            #                 'mag-nmlsquareasync0.0multiproble2probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.0multiproble5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.0multiproble10probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.0multiproble10probedynaratiosavemetricseq-probe-None-None+c4-2000',

            #                 # 对比一下不加calib
            #                 'mag-nmlmultiprobe5probesavemetricseq-probe-None-None+c4-2000',
            #                 'mag-nmlmultiprobe2probesavemetricseq-probe-None-None+c4-2000',
   
            #                 # 验证当前的epoch multiprobe
            #                 'mag-fullinfprobe-probe-None-None+c4-2000',
            #                 'mag-nmlmultiprobe10probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlmultiprobe5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlmultiprobe2probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlmultiprobe1probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 ],

            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            pass
        elif 'csr' in data:

            control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.3', '0.6', '0.8'], 
                             [
                            #      'mag-calib-probe-None-None+c4-2000',
                            #   'mag-calibrunningmean-probe-None-None+c4-2000',
                            # 'mag-calibnoqkema0.99-probe-None-None-fill+c4-2000',
                            # 'mag-calibnoqk-probe-None-None-each+c4-2000',
                            # 'mag-calibnoqkema0.99-probe-None-None-each+c4-2000',

                            # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe5each+c4-2000',
                            # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe1each+c4-2000',
                            # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe1None-probe1None-probe1each+c4-2000',

                            # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe10each+c4-2000',

                            # 'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe1None-probe1None-probe10each+c4-2000',

                            'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe1None-probe1None-probe1each+c4-2000',
                            'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe2None-probe2None-probe2each+c4-2000',
                            'mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe10each+c4-2000',
                            # 'mag-calibema0.99-probe-None-None-whole+c4-2000',

                            # 'mag-calibema0.99noqk-probe-None-None-fill+c4-2000',
                            # 'mag-calibnoqk-probe-None-None-fill+c4-2000',
                            # 'mag-calibnoqk-probe-None-None-each+c4-2000',

                            # 'mag-globalratiostdcalibema0.99noqk-probe-None-None-fill+c4-2000',
                            # 'mag-calibema0.99noqk-probe-None-None-each+c4-2000',
                            # 'mag-calibema0.99noqk-probe-None-None-whole+c4-2000',
                            # 'mag-savemetricseqcalibema0.99-probe-None-None+c4-2000',
                            # 'mag-globalratiostdcalibema0.99-probe-None-None+c4-2000',
                            
                            ],
        
                            ['q-proj+k-proj+v-proj+o-proj']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
            #  'arc-e', 'obqa-main''boolq', 'piqa', 'winogrande',
            control_name = [[[ 'arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], [ '0.7'], 
                             [
                            #      'mag-calib-probe-None-None+c4-2000',
                            #   'mag-calibrunningmean-probe-None-None+c4-2000',
                            'mag-calibema0.99-probe-None-None-None+c4-2000',
                            # 'mag-nmlprobe-probe-None-None+c4-2000',
                            # 'mag-similarityprobedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

                            # async probe with full inf fill and no clib (totally last round)
                            # 'mag-nmlhalfsquareasync0.0multiproble10probesavemetricseq-probe-None-None+c4-2000',
                            # 'mag-nmlhalfsquareasync0.0multiproble10probesavemetricseqema0.99calib-probe-None-None+c4-2000',

                            # # async probe with full inf fill
                            # 'mag-nmlhalfsquareasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
                            'mag-nmlhalfsquareasync0.3probedynaratiosavemetricseqema0.99calib-probe-None-None-None+c4-2000',

                            # async probe with different momentum
                            # 'mag-nmlsquareasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
                            # 'mag-nmlsquareasync0.3probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
                            # 'mag-nmlsquareasync0.5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
                            # 'mag-nmlsquareasync0.8probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

                            # # async multi probe
                            # 'mag-nmlsquareasync0.0multiproble2probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
                            # 'mag-nmlsquareasync0.0multiproble5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
                            # 'mag-nmlsquareasync0.0multiproble10probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
                            # 'mag-nmlsquareasync0.0multiproble10probedynaratiosavemetricseq-probe-None-None+c4-2000',

                            # # 对比一下不加calib
                            # 'mag-nmlmultiprobe5probesavemetricseq-probe-None-None+c4-2000',
                            # 'mag-nmlmultiprobe2probesavemetricseq-probe-None-None+c4-2000',
   
                            # # 验证当前的epoch multiprobe
                            # 'mag-fullinfprobe-probe-None-None+c4-2000',
                            # 'mag-nmlmultiprobe10probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
                            # 'mag-nmlmultiprobe5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
                            # 'mag-nmlmultiprobe2probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
                            'mag-nmlmultiprobe1probedynaratiosavemetricseqema0.99calib-probe-None-None-None+c4-2000',
                            ],
                            #   'mag-globalratiostdcalib-probe-None-None+c4-2000',
                            #   'mag-nmlprobe0.9calib-probe-None-None+c4-2000',
                            #   'mag-globalratiostdnmlprobe0.9calib-probe-None-None+c4-2000',
                            #   'mag-globalratiostdsavemetricseqv2rationmlprobe0.9calib-probe-None-None+c4-2000',
                            #   'mag-globalratiostdsavemetricseqv2rationmlprobe0.9calibrunningmean-probe-None-None+c4-2000'],
                            ['gate-proj+up-proj+down-proj']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
            # control_name = [[['arc-e'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], 
            #                  [
            #                 #      'mag-calib-probe-None-None+c4-2000',
            #                 #   'mag-calibrunningmean-probe-None-None+c4-2000',
            #                 'mag-calibema0.99-probe-None-None+c4-2000',
            #                 # 'mag-nmlprobe-probe-None-None+c4-2000',
            #                 # 'mag-similarityprobedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

            #                 # async probe with full inf fill and no clib (totally last round)
            #                 'mag-nmlhalfsquareasync0.0multiproble10probesavemetricseq-probe-None-None+c4-2000',
            #                 'mag-nmlhalfsquareasync0.0multiproble10probesavemetricseqema0.99calib-probe-None-None+c4-2000',

            #                 # async probe with full inf fill
            #                 'mag-nmlhalfsquareasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlhalfsquareasync0.3probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

            #                 # async probe with different momentum
            #                 'mag-nmlsquareasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.3probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.8probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',

            #                 # async multi probe
            #                 'mag-nmlsquareasync0.0multiproble2probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.0multiproble5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.0multiproble10probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlsquareasync0.0multiproble10probedynaratiosavemetricseq-probe-None-None+c4-2000',

            #                 # 对比一下不加calib
            #                 'mag-nmlmultiprobe5probesavemetricseq-probe-None-None+c4-2000',
            #                 'mag-nmlmultiprobe2probesavemetricseq-probe-None-None+c4-2000',
   
            #                 # 验证当前的epoch multiprobe
            #                 'mag-fullinfprobe-probe-None-None+c4-2000',
            #                 'mag-nmlmultiprobe10probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlmultiprobe5probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlmultiprobe2probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 'mag-nmlmultiprobe1probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 ],
            #                 #   'mag-globalratiostdcalib-probe-None-None+c4-2000',
            #                 #   'mag-nmlprobe0.9calib-probe-None-None+c4-2000',
            #                 #   'mag-globalratiostdnmlprobe0.9calib-probe-None-None+c4-2000',
            #                 #   'mag-globalratiostdsavemetricseqv2rationmlprobe0.9calib-probe-None-None+c4-2000',
            #                 #   'mag-globalratiostdsavemetricseqv2rationmlprobe0.9calibrunningmean-probe-None-None+c4-2000'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)





            # control_name = [[['arc-e'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.6'], 
            #                  [
                                 
            #                 #      'mag-calib-probe-None-None+c4-2000',
            #                 #   'mag-calibrunningmean-probe-None-None+c4-2000',
            #                 # 'mag-calibema0.9-probe-None-None+c4-2000',
            #                 # 'mag-calibema0.99-probe-None-None+c4-2000',
            #                 # 'mag-calibema0.999-probe-None-None+c4-2000'
            #                 'mag-nmlprobedynaratiosavemetricseqfillpbmetricoriginalema0.99calib-probe-None-None+c4-2000',
            #                 # 'mag-nmlprobedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000',
            #                 ],
            #                 #   'mag-globalratiostdcalib-probe-None-None+c4-2000',
            #                 #   'mag-nmlprobe0.9calib-probe-None-None+c4-2000',
            #                 #   'mag-globalratiostdnmlprobe0.9calib-probe-None-None+c4-2000',
            #                 #   'mag-globalratiostdsavemetricseqv2rationmlprobe0.9calib-probe-None-None+c4-2000',
            #                 #   'mag-globalratiostdsavemetricseqv2rationmlprobe0.9calibrunningmean-probe-None-None+c4-2000'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa-main'], ['llama-2-7b'], ['csr'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probe-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa-main'], ['llama-2-7b'], ['csr'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probe-fill-each', 'mag-probe-fill-each-delseq'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-e'], ['llama-2-7b'], ['csr'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probembsz-fill-each', 'mag-probe-fill-each-delseq', 'mag-probembsz-fill-each-delseq'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-e'], ['llama-2-7b'], ['csr'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probembsz-fill-each', 'mag-probe-fill-each', 'mag-probembsz-fill-each-delseq'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-e'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probe-None-None', 'mag-probembsz-None-None', 'mag-probembszmseq-None-None'],
            #         ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-e'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probembsz-fill-each-onlyvo', 'mag-probembsz-fill-each-delseq-onlyvo'],
            #         ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-e'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-probembszwandasp-fill-each', 'mag-probembszwandasp-fill-each-delseq'],
            #         ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            pass
    elif file == 'test_dense_model':
        controls = []
        script_name = [[f'{filename}.py']]
        if 'clm' in data:

            control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b'], ['clm'], ['10'], ['128'], ['0'], 
                             ['None'], ['dense'], ['sync'], ['None'], ['None'],        
                            ['None']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0'], ['dense'],
            #         ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-6.7b', 'opt-13b'], ['clm'], ['10'], ['128'], ['0'], ['dense'],
            #         ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['512', '1024'], ['0'], ['dense'],
            #         ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['2048'], ['0'], ['dense'],
            #         ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], [ 'opt-13b'], ['clm'], ['10'], ['128', '512', '1024'], ['0'], ['dense'],
            #         ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], [ 'llama-2-7b'], ['clm'], ['10'], ['128', '512', '1024'], ['0'], ['dense'],
            #         ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            pass
        elif 'csr' in data:
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa-main'], ['llama-2-7b', 'llama-2-13b', 'llama-2-70b'], ['csr'], ['10'], ['128'], ['0'], ['dense'],
            #         ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c',  'obqa-main'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0'], ['dense'],
            #         ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            control_name = [[[ 'arc-c',  'arc-e'], ['llama-2-7b'], ['csr'], ['10'], ['1024'], ['0'], ['dense'],
                    ['None']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
            pass
    elif file == 'test_fix_pruned_model': 
        controls = []
        script_name = [[f'{filename}.py']]
        if 'clm' in data:
            control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b'], ['clm'], ['10', '50', '100'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-wandasp+128'],
                    ['o-proj+down-proj']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b'], ['clm'], ['10'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-wandasp+128', 'mag-flap+128'],
            #         ['down-proj', 'o-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b'], ['clm'], ['10'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-wandasp+128', 'mag-flap+128'],
            #         ['o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b'], ['clm'], ['50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-wandasp+128', 'mag-flap+128'],
            #         ['o-proj', 'down-proj', 'o-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-wandasp-maintain+128', 'mag-flap-maintain+128'],
            #         ['o-proj', 'down-proj', 'o-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-wandasp-cascadeattn+128', 'mag-flap-cascadeattn+128'],
            #         ['o-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b'], ['clm'], [ '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], ['mag-wandasp+128', 'mag-flap+128'],
            #         ['o-proj','down-proj', 'o-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            pass
        elif 'csr' in data:
            # control_name = [[['boolq', 'piqa', 'arc-e', 'arc-c', 'hellaswag', 'winogrande', 'obqa-main'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], [ 'mag-wandasp+128','mag-flap+128'],
            #         ['o-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-e', ], ['llama-2-7b'], ['csr'], ['10', '50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], [ 'mag-wandasp+128','mag-flap+128'],
            #         ['o-proj', 'down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-e', ], ['llama-2-7b'], ['csr'], ['50'], ['128'], ['0.1', '0.2', '0.3', '0.4', '0.5'], [ 'mag-wandasp+128','mag-flap+128'],
            #         ['o-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            pass
            
        elif 'missing' in data:
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode)
            controls.extend(CIFAR10_controls_9)
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

    def delete_file_if_exist(file_name):
        if os.path.exists(file_name):
        # Delete the file if it exists
            os.remove(file_name)
    # Check if the file exists
    
    delete_file_if_exist(bash_file_name)

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
        print('controls[i]', controls[i])
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

            if 'llama-2-7b' in controls[i][4] or 'opt-6.7b' in controls[i][4] or 'opt-1.3b' in controls[i][4]:
                run_time = '00:45:00'
            elif 'llama-2-13b' in controls[i][4] or 'opt-13b' in controls[i][4]:
                run_time = '01:45:00'
            elif 'llama-2-70b' in controls[i][4]:
                run_time = '03:45:00'
            j += 1
            i += 1
            
        
        # print('isgpt', is_gpt)
        temp_mem = mem
        if is_llama:
            if '7b' in controls[i-1][4] or '6.7b' in controls[i-1][4] or '1.3b' in controls[i-1][4]:
                temp_mem = int(2.5 * mem)
            elif '13b' in controls[i-1][4]:
                temp_mem = int(4.3 * mem)
            elif '70b' in controls[i-1][4]:
                temp_mem = int(13.5 * mem)
            # temp_mem = int(3.5 * mem)
        if is_opt:
            temp_mem = int(1.5 * temp_mem)
        if is_gpt:
            temp_mem = int(1.5 * temp_mem)
        s = '#!/bin/bash -l\n'
        # s += f'#SBATCH --time={run_time}\n'
        # s += f'#SBATCH --nodes={task_parallel_num}\n'
        # s += f'#SBATCH --ntasks={task_parallel_num}\n'
        # # s += '#SBATCH --cpus-per-task=2'
        # s += '#SBATCH --gres=gpu:a100:1\n'
        # s += '#SBATCH --partition=a100-4\n'
        # s += f'#SBATCH --mem={temp_mem}gb\n'
        # # s += '#SBATCH --mail-type=ALL \n'
        # # s += '#SBATCH --mail-user=le000288@umn.edu\n'
        # s += f'#SBATCH -o {res_path}/{filename}_%j.out\n'
        # s += f'#SBATCH -e {res_path}/{filename}_%j.err\n'
        # s += '\n'
        # s += f'cd /home/aanwar/le000288/{code_folder}/src\n'
        # s += '\n'
        # s += 'export PATH=/home/aanwar/le000288/miniconda3/envs/eri/bin:$PATH\n'
        # if 'max' in controls[i][-1]:
        #     s_for_max = s_for_max + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
        #         '--resume_mode {} --log_interval {} --device {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])
        s += 'timestamp=$(date +%Y%m%d%H%M%S)'

        #     if k_for_max % round == round - 1:
        #         s_for_max = s_for_max[:-2] + '\nwait\n'
        #     k_for_max = k_for_max + 1
        #     continue
        # while i < len(controls):
        # srun --nodes=1 --ntasks=1 
        # time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        for item in sub_controls:
            s += '\n'
            s = s + 'python {} --device {} --resume_mode {} --init_seed {} --control_name {} &> wslout/output_{}_$timestamp.txt\n'.format(*item, item[-1])

        s += 'wait\n'
        # controls[i][0] = 'test_classifier_fl.py'
        # for item in sub_controls:
        #     item[0] = item[0].replace('train', 'test')
        #     print(item, item[0])
        #     s += '\n'
        #     s = s + 'srun --nodes=1 --ntasks=1 python {} --device {} --resume_mode {} --init_seed {} --control_name {}&\n'.format(*item)
        # s += 'wait\n'
        pbs_file_name = './{}.sh'.format(f'{filename}')
        # Check if the file exists
        if os.path.exists(pbs_file_name):
            # Delete the file if it exists
            os.remove(pbs_file_name)
        run_file = open(pbs_file_name, 'a')
        run_file.write(s)
        run_file.close()

        run_file = open(bash_file_name, 'a')
        command = f'bash {filename}.sh --wait\n'
        run_file.write(command)
        run_file.close()

        with open(bash_file_name, 'r') as cur_file:
            line_count = sum(1 for line in cur_file)

        if line_count > 180:
            bash_file_name = './{}.sh'.format(f'msi_{file}_{data[0]}_{i}')
            print('bash_file_name', bash_file_name)
            delete_file_if_exist(bash_file_name)
    return


if __name__ == '__main__':
    main()
