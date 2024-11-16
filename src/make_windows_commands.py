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
    print('gpu_ids', gpu_ids)
    # return 
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
    

    if file == 'test_model':
        controls = []
        script_name = [[f'{filename}.py']]
        if 'clm' in data:
            print('here')
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['128'], ['0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-20'], ['0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-respick'], ['sync'], ['None'], ['0.1-0.1-0.1-0.1-0.1-seqrank', '1-1-1-1-1-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-6.7b'], ['clm'], ['10'], ['128'], ['0.4'], 
            #                  ['ppwandasp'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-6.7b'], ['clm'], ['10'], ['128'], ['0.4'], 
            #                  ['ppwandasp'], ['calib-ema'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-6.7b'], ['clm'], ['10'], ['128'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-3-8b'], ['clm'], ['5'], ['128', '2048'], ['0.2'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-20'], ['None'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b'], ['clm'], ['10'], ['120'], ['0.4'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-20'], ['None'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.4'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-20'], ['None'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['1'], ['128'], ['0.4'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-20'], ['None'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['1'], ['128'], ['0.4'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-20'], ['None'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.2-0.6', '0.4-0.6', '0.6', '0.5'], 
            #                  ['ppwandasp'], ['calib'], ['asyncinter'], ['c4-20'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['5'], ['128'], ['0.2'], 
            #                  ['wandasp', 'flap'], ['calib-resinfo'], ['asyncinter'], ['c4-20'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['5'], ['128'], ['0.6'], 
            #                  ['wandasp'], ['probe-default-recorddiff'], ['sync'], ['c4-20'], ['0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-3-8b'], ['clm'], ['2'], ['128'], ['0.4'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.4'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.4'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.5'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-100'], ['0.1-0.1-0.1-0.1-0.1-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.5'], 
            #                  ['ppwandasp'], ['probe-default-shengxiaseq'], ['sync'], ['c4-100'], ['0.1-0.1-0.1-0.1-0.1-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.5'], 
            #                  ['wandasp'], ['probe-default'], ['sync'], ['c4-100'], ['0.1-0.1-0.1-0.1-0.1-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.5'], 
            #                  ['wandasp'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.5'], 
            #                  ['ppwandasp'], ['probe'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.6'], 
            #                  ['flap'], ['calib', 'calib-ema'], ['asyncinter'], ['c4-200'], ['None'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.6'], 
            #                  ['flap'], ['calib-ema-probe'], ['asyncintra'], ['c4-200'], ['seqrank+bszrank-0-0-0-0.1+0.1-0.1+0.1', 'bszrank-0-0-0-0.1-0.1', 'seqrank-0-0-0-0.1-0.1'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0.6', '0.8'], 
            #                  ['wandasp', 'flap', 'probe'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['512'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.05-0.05-0.05-0.05-0.05-seqrank', '0.9-0.9-0.9-0.9-0.9-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['512'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.05+0.05-0.05+0.05-0.05+0.05-0.05+0.05-0.05+0.05-seqrank+bszrank', '0.05+0.05-0.05+0.05-0.05+0.05-0.05+0.05-0.05+0.05-bszrank+seqrank', '0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-6.7b'], ['clm'], ['20'], ['512'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default', 'probe-default-randomrank', 'probe-default-rulerank-last', 'probe-rulerank', 'probe-respick'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.05+0.5-0.05+0.5-0.05+0.5-0.05+0.5-0.05+0.5-bszrank+seqrank', '0.05-0.05-0.05-0.05-0.05-seqrank', '0.05+0.05-0.05+0.05-0.05+0.05-0.05+0.05-0.05+0.05-bszrank+seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['512'], ['0.4'], 
            #                 ['ppwandasp'], ['probe-default'], ['asyncintra'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #             ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-3.2-3b'], ['clm'], ['10'], ['512'], ['0.4'], 
                            ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
                        ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['256'], ['0.4'], 
                            ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
                        ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
            pass
        elif 'csr' in data:

            # control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.5'], 
            #                  ['wandasp'], ['calib-ema-probe'], ['sync'], ['c4-200'], ['0.1normwhole4-0.1normwhole4-0.1normwhole4-None-None-seqrank'],
            #                 ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.5'], 
            #                  ['wandasp'], ['calib-ema'], ['asyncinter'], ['c4-200'], ['0.1normwhole4-0.1normwhole4-0.1normwhole4-None-None'],
            #                 ['q-proj+k-proj+v-proj+o-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.5'], 
            #                  ['wandasp'], ['calib-ema'], ['asyncinter'], ['c4-2000'], ['0.1normwhole4-0.1normwhole4-0.1normwhole4-0.1norm4-0.1norm4'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.5'], 
            #                  ['wandasp'], ['calib-ema-probe-norm'], ['sync'], ['c4-2000'], ['0.1normwhole4-0.1normwhole4-0.1normwhole4-0.1norm4-0.1norm4-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['obqa-main'], ['llama-2-7b'], ['csr'], ['10'], ['128', '1024'], ['0.5'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-bszrank', '0.1-0.1-0.1-0.1-0.1-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['obqa-main', 'arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.5'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-20'], ['0.1-0.1-0.1-0.1-0.1-bszrank', '0.1-0.1-0.1-0.1-0.1-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['obqa-main', 'arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.5'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-20'], ['0.2-0.2-0.2-0.2-0.2-bszrank', '0.5-0.5-0.5-0.5-0.5-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            
            # control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.5'], 
            #                  ['ppwandasp'], ['probe'], ['sync'], ['c4-20'], ['0.2-0.2-0.2-0.2-0.2-bszrank', '0.5-0.5-0.5-0.5-0.5-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.5'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-200'], ['0.2-0.2-0.2-0.2-0.2-bszrank', '0.5-0.5-0.5-0.5-0.5-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.5'], 
            #                  ['ppwandasp'], ['calib', 'calib-ema'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['1024'], ['0.5'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-200'], ['0.1-0.1-0.1-0.1-0.1-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['10'], ['128'], ['0.5'], 
            #                  ['ppwandasp'], ['probe-calib-ema'], ['sync'], ['c4-200'], ['0.1-0.1-0.1-0.1-0.1-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-3-8b', 'llama-2-7b'], ['clm'], ['10'], ['128'], ['0.2'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-bszrank'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['1'], ['512'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['obqa', 'arc-c'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-calib-respick', 'probe-calib-respick-ema', 'probe-calib-respick-rightpad'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-seqrank', '0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.05+0.5-0.05+0.5-0.05+0.5-0.05+0.5-0.05+0.5-bszrank+seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            
            # control_name = [[['obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['asyncintra'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            pass
        elif 'mix' in data:
            control_name = [[['arc-c+wikitext-2v1'], ['llama-2-7b'], ['mix'], ['20'], ['256'], ['0.2', '0.4'], 
                            ['ppwandasp'], ['probe-default'], ['sync'], ['c4-20'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
                        ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
    elif file == 'save_calib_info':
        controls = []
        script_name = [['test_model.py']]
        if 'clm' in data:
            control_name = [[['wikitext-2v1', 'ptb'], ['llama-3-8b', 'llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2'], 
                             ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],        
                            ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            # 'llama-3-8b', 'llama-2-7b', 'llama-2-13b', 'opt-13b'
            control_name = [[['wikitext-2v1', 'ptb'], ['llama-3-8b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2'], 
                             ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],        
                            ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1', 'ptb'], ['llama-3-8b', 'llama-2-7b'], ['clm'], ['10'], ['128'], ['0.2'], 
                             ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-bszrank'],        
                            ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-30b'], ['clm'], ['20'], ['2048'], ['0'], 
            #                  ['None'], ['dense'], ['None'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            pass
        elif 'csr' in data:
            control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b', 'llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2'], 
                             ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
                            ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b', 'llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2'], 
                             ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
                            ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b', 'llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2'], 
                             ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-bszrank'],
                            ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-30b'], ['csr'], ['20'], ['512'], ['0.2'], 
            #                  ['flap', 'wandasp'], ['flap-default', 'wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-30b'], ['csr'], ['20'], ['512'], ['0.2'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            pass
    elif file == 'test_dense_model':
        controls = []
        script_name = [[f'{filename}.py']]
        if 'clm' in data:
            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['512'], ['0'], 
                             ['None'], ['dense'], ['None'], ['None'], ['None'],        
                            ['None']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['1024'], ['0'], 
            #                  ['None'], ['dense'], ['None'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['opt-6.7b'], ['clm'], ['10'], ['128'], ['0'], 
            #                  ['None'], ['dense'], ['None'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-3-8b'], ['clm'], ['10'], ['128'], ['0'], 
            #                  ['None'], ['dense'], ['None'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['c4'], ['llama-2-7b'], ['clm'], ['10'], ['128'], ['0'], 
            #                  ['None'], ['dense'], ['None'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['10'], ['896'], ['0'], 
            #                  ['None'], ['dense'], ['sync'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

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
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'llama-3-8b'], ['csr'], ['20'], ['512'], ['0'], 
            #                  ['None'], ['dense'], ['None'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            control_name = [[['obqa'], ['llama-2-7b', 'llama-3-8b'], ['csr'], ['20'], ['512'], ['0'], 
                             ['None'], ['dense'], ['None'], ['None'], ['None'],        
                            ['None']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
            pass
    elif file == 'test_local_tuned_model': 
        controls = []
        script_name = [[f'{filename}.py']]
        if 'clm' in data:
            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
                             ['None'], ['llmpruner-prune', 'llmpruner-tune'], ['asyncinter'], ['None'], ['None'],
                            ['None']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
                             ['None'], ['loraprune-prune', 'loraprune-tune'], ['asyncinter'], ['None'], ['None'],
                            ['None']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
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


            control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
                             ['None'], ['llmpruner-prune', 'llmpruner-tune'], ['asyncinter'], ['None'], ['None'],
                            ['None']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
                             ['None'], ['loraprune-prune', 'loraprune-tune'], ['asyncinter'], ['None'], ['None'],
                            ['None']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
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

    bash_file_name = './{}.bash'.format(f'windows_{file}_{data[0]}')

    def delete_file_if_exist(file_name):
        if os.path.exists(file_name):
        # Delete the file if it exists
            os.remove(file_name)
    # Check if the file exists
    
    delete_file_if_exist(bash_file_name)

    task_parallel_num = 1
    mem = 15
    if task_parallel_num == 1:
        mem = 15
    elif task_parallel_num == 2:
        mem = 45
    elif task_parallel_num == 3:
        mem = 65
    

    i = 0
    kk = 0
    while i < len(controls):
    # for i in range(len(controls)):
        controls[i] = list(controls[i])
        # print('controls[i]', controls[i])
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
            # print('controls[i]', controls[i])
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
            # s = s + 'CUDA_VISIBLE_DEVICES={} python {} --device {} --resume_mode {} --init_seed {} --control_name {} -- &> wslout/output_{}_$timestamp.txt\n'.format(gpu_ids[kk % len(gpu_ids)], *item, item[-1])
            # print('CUDA_VISIBLE_DEVICES={} python {} --device {} --resume_mode {} --init_seed {} --control_name {} -- &> wslout/output_{}_$timestamp.txt\n'.format(gpu_ids[k % len(gpu_ids)], *item, item[-1]))
        kk += 1

  
            # s = s + 'python {} --device {} --resume_mode {} --init_seed {} --control_name {}\n'.format(*item)


            # s = s + 'nsys profile -w true --gpu-metrics-device=0 -x true --force-overwrite=true -o {} python {} --device {} --resume_mode {} --init_seed {} --control_name {} &> wslout/output_{}_$timestamp.txt\n'.format(item[4], *item, item[-1])
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
        if kk % len(gpu_ids) == 0:
            command = f'bash {filename}.sh\n'
            # command = f'wait\n'
        else:
            command = f'bash {filename}.sh\n'
        run_file.write(command)
        
        # run_file.write('sleep 20\n')
        run_file.close()

        with open(bash_file_name, 'r') as cur_file:
            line_count = sum(1 for line in cur_file)

        if line_count > 180:
            bash_file_name = './{}.sh'.format(f'windows_{file}_{data[0]}_{i}')
            print('bash_file_name', bash_file_name)
            delete_file_if_exist(bash_file_name)
            
        # s += 'wait\n'
        # # controls[i][0] = 'test_classifier_fl.py'
        # # for item in sub_controls:
        # #     item[0] = item[0].replace('train', 'test')
        # #     print(item, item[0])
        # #     s += '\n'
        # #     s = s + 'srun --nodes=1 --ntasks=1 python {} --device {} --resume_mode {} --init_seed {} --control_name {}&\n'.format(*item)
        # # s += 'wait\n'
        # pbs_file_name = './{}.sh'.format(f'{filename}')
        # # Check if the file exists
        # if os.path.exists(pbs_file_name):
        #     # Delete the file if it exists
        #     os.remove(pbs_file_name)
        # run_file = open(pbs_file_name, 'a')
        # run_file.write(s)
        # run_file.close()

        # run_file = open(bash_file_name, 'a')
        # if kk % len(gpu_ids) == 0:
        #     command = f'bash {filename}.sh\n'
        #     command = f'wait\n'
        # else:
        #     command = f'bash {filename}.sh &\n'
        # run_file.write(command)
        
        # # run_file.write('sleep 20\n')
        # run_file.close()

        # with open(bash_file_name, 'r') as cur_file:
        #     line_count = sum(1 for line in cur_file)

        # if line_count > 180:
        #     bash_file_name = './{}.sh'.format(f'windows_{file}_{data[0]}_{i}')
        #     print('bash_file_name', bash_file_name)
        #     delete_file_if_exist(bash_file_name)
    return


if __name__ == '__main__':
    main()
