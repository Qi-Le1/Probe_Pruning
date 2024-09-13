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

# first run dense model, get dense running time and flops
# second run 1 2000c4 calib for flap/wandasp/ppwandasp for each model to store calibration info
# then run all exps

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
            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-13b'], ['clm'], ['20'], ['1024'], [ '0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-seqrank', '0.1-0.1-0.1-0.1-0.1-bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank', '0.05-0.05-0.05-0.05-0.05-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # llama-3-8b
            # control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            
            # comparing metric
            # control_name = [[['wikitext-2v1'], ['llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.6'], 
            #                  ['wandasp', 'ppwandasp', 'flap'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['q-proj+k-proj+v-proj+o-proj', 'gate-proj+up-proj+down-proj', 'default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            
            # control_name = [[['wikitext-2v1'], ['opt-13b'], ['clm'], ['20'], ['1024'], ['0.6'], 
            #                  ['wandasp', 'ppwandasp', 'flap'], ['calib'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['q-proj+k-proj+v-proj+out-proj', 'fc1+fc2', 'default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # comparing different calibration dataset for FLAP
            # control_name = [[['wikitext-2v1'], ['llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['wikivalid-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # comparing square or not for flap
            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['flap'], ['flap-default', 'flap-default-square'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # optimal probe
            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe'], ['sync'], ['None'], ['1-1-1-1-1-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe'], ['sync'], ['None'], ['1-1-1-1-1-bszrank'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # inorder wiki
            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default-inorderwiki'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # different attention mlp ratio
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0-0', '0-0.2', '0-0.4', '0-0.6', '0.2-0', '0.2-0.2', '0.2-0.4', '0.2-0.6', '0.4-0', '0.4-0.2', '0.4-0.4', '0.4-0.6', '0.6-0', '0.6-0.2', '0.6-0.4', '0.6-0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # probe can increase all metric performance
            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['wandasp', 'flap'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # get res similarity with output
            # control_name = [[['wikitext-2v1'], ['llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.4', '0.6'], 
            #                  ['ppwandasp', 'wandasp'], ['calib-resinfo0.7', 'calib-resinfo0.9', 'calib-resinfo1'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # asyncintra
            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['asyncintra'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # calib / calib + ema / only probe / probe + ema
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['calib', 'calib-ema'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-respick'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank', '0.1-0.1-0.1-0.1-0.1-seqrank', '0.15-0.15-0.15-0.15-0.15-seqrank',\
            #                                                                                        '0.2-0.2-0.2-0.2-0.2-seqrank', '0.05-0.05-0.05-0.05-0.05-bszrank', '0.1-0.1-0.1-0.1-0.1-bszrank', '0.15-0.15-0.15-0.15-0.15-bszrank',\
            #                                                                                        '0.2-0.2-0.2-0.2-0.2-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # probe ratio bianhua (zanshimeiyong)
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.05-0.05-0.05-0.05-0.05-seqrank', '0.1-0.1-0.1-0.1-0.1-seqrank', '0.15-0.15-0.15-0.15-0.15-seqrank',\
            #                                                                                        '0.2-0.2-0.2-0.2-0.2-seqrank', '0.05-0.05-0.05-0.05-0.05-bszrank', '0.1-0.1-0.1-0.1-0.1-bszrank', '0.15-0.15-0.15-0.15-0.15-bszrank',\
            #                                                                                        '0.2-0.2-0.2-0.2-0.2-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], [ '0.3-0.3-0.3-0.3-0.3-seqrank', '0.4-0.4-0.4-0.4-0.4-seqrank', '0.5-0.5-0.5-0.5-0.5-seqrank', '0.6-0.6-0.6-0.6-0.6-seqrank', '0.7-0.7-0.7-0.7-0.7-seqrank', '0.8-0.8-0.8-0.8-0.8-seqrank', '0.9-0.9-0.9-0.9-0.9-seqrank', \
                                                                                                   
            #                                                                                        '0.3-0.3-0.3-0.3-0.3-bszrank', '0.4-0.4-0.4-0.4-0.4-bszrank', '0.5-0.5-0.5-0.5-0.5-bszrank', '0.6-0.6-0.6-0.6-0.6-bszrank', '0.7-0.7-0.7-0.7-0.7-bszrank', '0.8-0.8-0.8-0.8-0.8-bszrank', '0.9-0.9-0.9-0.9-0.9-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # probevsprobenocalib
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['5'], ['1024'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-respick', 'probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # inferencespeed
            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #              ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #             ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2',  '0.4', '0.6'], 
            #                 ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # no respick
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-calib-ema'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # probe+calib fix ratio
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default-probefixratio0.5', 'probe-default-probefixratio0.9'], ['sync'], ['c4-2000'], 
            #                  ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # different probe effect
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], 
            #         [ '0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank', '0.1-0.1-0.1-0.1-0.1-seqrank', 
            #                                                                                        '0.2-0.2-0.2-0.2-0.2-seqrank', '0.05-0.05-0.05-0.05-0.05-bszrank', '0.1-0.1-0.1-0.1-0.1-bszrank',
            #                                                                                        '0.2-0.2-0.2-0.2-0.2-bszrank'],
            #                 ['q-proj+k-proj+v-proj+o-proj', 'gate-proj+up-proj+down-proj', 'default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # record select channel diff
            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default-recorddiff'], ['sync'], ['c4-2000'], 
            #         [ '0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # -------- 7/31/2024
            # test middle block probe effect
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe'], ['sync'], ['None'], ['1-1-1-1-1-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-calib-respick'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #              ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #             ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['1'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.05-0.05-0.05-0.05-0.05-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.6'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.6'], 
            #                  ['flap'], ['probe-flap-default'], ['sync'], ['c4-2000'], ['1-1-1-1-1-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-calib-respick-skip-0-4'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['calib-respick-skip-0-4'], ['asyncinter'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-calib-respick-skip-0-26'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['calib-respick-skip-0-26'], ['asyncinter'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4'], 
            #                  ['ppwandasp'], ['calib-respick-skip-0-31'], ['asyncinter'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['ptb', 'wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4', '0.6'], 
            #                 ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank'],
            #             ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.4', '0.6'], 
            #                 ['ppwandasp'], ['probe-calib-respick-ema-ruleranklast', 'probe-calib-respick-ema'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank'],
            #             ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['512'], ['0.4', '0.6'], 
            #                 ['ppwandasp'], ['probe-calib-respick-ema-ruleranklast', 'probe-calib-respick-ema', 'probe-calib-respick-ema-mixing'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank'],
            #             ['q-proj+k-proj+v-proj+o-proj', 'gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['512'], ['0.4'], 
            #                 ['ppwandasp'], ['probe-calib-respick-ema-deleteoutlier'], ['sync'], ['c4-2000'], ['0.05-0.05-0.05-0.05-0.05-seqrank'],
            #             ['q-proj+k-proj+v-proj+o-proj', 'gate-proj+up-proj+down-proj', 'default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['512'], ['0.4', '0.6'], 
            #                 ['ppwandasp'], ['probe-respick-calib-ema-deleteoutlier'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #             ['q-proj+k-proj+v-proj+o-proj', 'gate-proj+up-proj+down-proj', 'default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['512'], ['0.4'], 
            #                 ['ppwandasp'], ['probe-respick-calib-ema-deleteoutlier'], ['sync'], ['c4-2000'], ['0.05+0.05-0.05+0.05-0.05+0.05-0.05+0.05-0.05+0.05-seqrank+bszrank', '0.05+0.05-0.05+0.05-0.05+0.05-0.05+0.05-0.05+0.05-bszrank+seqrank'],
            #             ['q-proj+k-proj+v-proj+o-proj', 'gate-proj+up-proj+down-proj', 'default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['10', '50', '512'], ['0'], 
                            ['ppwandasp'], ['probe-calib-respick-ema'], ['sync'], ['c4-2000'], ['0.05-0.05-0.05-0.05-0.05-seqrank'],
                        ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['5'], ['2048'], ['0'], 
                            ['ppwandasp'], ['probe-calib-respick-ema'], ['sync'], ['c4-2000'], ['0.05-0.05-0.05-0.05-0.05-seqrank'],
                        ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
            # pass
        
        elif 'csr' in data:
            
            # opt chongpao
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['opt-13b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-e', 'piqa', 'obqa', 'boolq',  'hellaswag', 'winogrande', 'arc-c'], ['opt-13b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['wandasp', 'flap'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['opt-13b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['asyncintra'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['csr'], ['20'], ['512'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-seqrank', '0.1-0.1-0.1-0.1-0.1-bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank', '0.05-0.05-0.05-0.05-0.05-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # llama-3-8b
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # optimal probe
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'opt-13b', 'llama-2-13b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe'], ['sync'], ['None'], ['1-1-1-1-1-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe'], ['sync'], ['None'], ['1-1-1-1-1-bszrank'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # different attention mlp ratio
            # control_name = [[['hellaswag', 'arc-c', 'arc-e'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0-0', '0-0.2', '0-0.4', '0-0.6', '0.2-0', '0.2-0.2', '0.2-0.4', '0.2-0.6', '0.4-0', '0.4-0.2', '0.4-0.4', '0.4-0.6', '0.6-0', '0.6-0.2', '0.6-0.4', '0.6-0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            
            # control_name = [[['boolq', 'piqa', 'winogrande',  'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0-0', '0-0.2', '0-0.4', '0-0.6', '0.2-0', '0.2-0.2', '0.2-0.4', '0.2-0.6', '0.4-0', '0.4-0.2', '0.4-0.4', '0.4-0.6', '0.6-0', '0.6-0.2', '0.6-0.4', '0.6-0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['piqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], [ '0-0.6', '0.2-0', '0.2-0.2', '0.2-0.4', '0.2-0.6', '0.4-0', '0.4-0.2', '0.4-0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # arcepiqaobqa probe can increase all metric performance
            # control_name = [[['arc-e', 'piqa', 'obqa', 'boolq',  'hellaswag', 'winogrande', 'arc-c'], ['llama-2-7b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['wandasp', 'flap'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

        

            # get resinfo
            # control_name = [[['arc-c'], ['llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.4', '0.6'], 
            #                  ['ppwandasp', 'wandasp'], ['calib-resinfo0.7', 'calib-resinfo0.9', 'calib-resinfo1'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            #asyncintra
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['asyncintra'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['asyncintra'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # calib / calib + ema 
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['calib', 'calib-ema'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # no respick
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-calib-ema'], ['asyncintra'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # probevsprobenocalib
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-respick'], ['sync'], ['c4-2000'], 
            #                  ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-respick'], ['sync'], ['c4-2000'], 
            #                  ['0.1-0.1-0.1-0.1-0.1-bszrank', '0.2-0.2-0.2-0.2-0.2-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], 
            #                  ['0.1-0.1-0.1-0.1-0.1-bszrank', '0.2-0.2-0.2-0.2-0.2-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['5'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-respick', 'probe-default'], ['sync'], ['c4-2000'], 
            #                  ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)



            # probe+calib fix ratio
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2','0.4', '0.6'], 
            #                  ['ppwandasp'], ['probe-default-probefixratio0.5', 'probe-default-probefixratio0.9'], ['sync'], ['c4-2000'], 
            #                  ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # different probe
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.6'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], 
            #         [ '0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank', '0.1-0.1-0.1-0.1-0.1-seqrank', 
            #                                                                                        '0.2-0.2-0.2-0.2-0.2-seqrank', '0.05-0.05-0.05-0.05-0.05-bszrank', '0.1-0.1-0.1-0.1-0.1-bszrank',
            #                                                                                        '0.2-0.2-0.2-0.2-0.2-bszrank'],
            #                 ['q-proj+k-proj+v-proj+o-proj', 'gate-proj+up-proj+down-proj', 'default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            # --- rebuttal
            # control_name = [[['winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['1'], ['512'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.05-0.05-0.05-0.05-0.05-seqrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-c'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.4'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.4', '0.6'], 
            #                 ['ppwandasp'], ['probe-calib-respick-ema-ruleranklast', 'probe-calib-respick-ema'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank', '0.05-0.05-0.05-0.05-0.05-seqrank'],
            #             ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)


            control_name = [[['arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.4'], 
                            ['ppwandasp'], ['probe-calib-respick-ema-dimmetric'], ['sync'], ['c4-2000'], ['0.05-0.05-0.05-0.05-0.05-seqrank'],
                        ['default']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            pass
    # gather calib info to save time
    # model, seq, metric, calib, target module important

    elif file == 'save_calib_info':
        controls = []
        script_name = [['test_model.py']]
        if 'clm' in data:
            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # # 'llama-3-8b', 'llama-2-7b', 'llama-2-13b', 'opt-13b'
            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1', 'ptb'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0.2'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-bszrank'],        
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # 'llama-3-8b'
            # control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0.2'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],        
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            
            # control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0.2'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],        
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0.2'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-bszrank'],        
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

             # 'llama-2-70b'
            control_name = [[['wikitext-2v1'], ['llama-2-70b'], ['clm'], ['20'], ['1024'], ['0.2'], 
                             ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],        
                            ['gate-proj+up-proj+down-proj']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
            
            control_name = [[['wikitext-2v1'], ['llama-2-70b'], ['clm'], ['20'], ['1024'], ['0.2'], 
                             ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],        
                            ['gate-proj+up-proj+down-proj']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-70b'], ['clm'], ['20'], ['1024'], ['0.2'], 
                             ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank'],        
                            ['gate-proj+up-proj+down-proj']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
            pass
        elif 'csr' in data:
            # control_name = [[['obqa'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.2'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['obqa'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.2'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['obqa'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0.2'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-bszrank'],
            #                 ['default']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # 'llama-3-8b'
            # control_name = [[['obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0.2'], 
            #                  ['flap'], ['flap-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0.2'], 
            #                  ['wandasp'], ['wandasp-default'], ['asyncinter'], ['c4-2000'], ['None'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # control_name = [[['obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0.2'], 
            #                  ['ppwandasp'], ['probe-default'], ['sync'], ['c4-2000'], ['0.1-0.1-0.1-0.1-0.1-bszrank'],
            #                 ['gate-proj+up-proj+down-proj']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            pass
    elif file == 'test_dense_model':
        controls = []
        script_name = [[f'{filename}.py']]
        if 'clm' in data:
            # 'llama-2-7b', 'llama-2-13b', 'opt-13b'
            # control_name = [[['wikitext-2v1'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['clm'], ['20'], ['1024'], ['0'], 
            #                  ['None'], ['dense'], ['None'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # llama-3-8b
            # control_name = [[['wikitext-2v1'], ['llama-3-8b'], ['clm'], ['20'], ['1024'], ['0'], 
            #                  ['None'], ['dense'], ['None'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # llama-2-70b
            # control_name = [[['wikitext-2v1'], ['llama-2-70b'], ['clm'], ['20'], ['1024'], ['0'], 
            #                  ['None'], ['dense'], ['None'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            # ---- 8/4/2024 break latency
            control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0'], 
                             ['None'], ['dense'], ['None'], ['None'], ['None'],        
                            ['None']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
            pass
        elif 'csr' in data:
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b', 'llama-2-13b', 'opt-13b'], ['csr'], ['20'], ['512'], ['0'], 
            #                  ['None'], ['dense'], ['None'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-3-8b'], ['csr'], ['20'], ['512'], ['0'], 
            #                  ['None'], ['dense'], ['None'], ['None'], ['None'],        
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)
            pass
        elif 'missing' in data:
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode)
            controls.extend(CIFAR10_controls_9)
    elif file == 'test_local_tuned_model': 
        controls = []
        script_name = [[f'{filename}.py']]
        if 'clm' in data:
            # control_name = [[['wikitext-2v1'], ['llama-2-7b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
            #                  ['None'], ['llmpruner-prune', 'llmpruner-tune'], ['asyncinter'], ['None'], ['None'],
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            control_name = [[['wikitext-2v1'], ['llama-2-13b'], ['clm'], ['20'], ['1024'], ['0.2', '0.4', '0.6'], 
                             ['None'], ['llmpruner-prune', 'llmpruner-tune'], ['asyncinter'], ['None'], ['None'],
                            ['None']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            controls.extend(CIFAR10_controls_9)
            pass
        elif 'csr' in data:
            # control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-7b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
            #                  ['None'], ['llmpruner-prune', 'llmpruner-tune'], ['asyncinter'], ['None'], ['None'],
            #                 ['None']]]
            # CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
            # controls.extend(CIFAR10_controls_9)

            control_name = [[['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc-c', 'arc-e', 'obqa'], ['llama-2-13b'], ['csr'], ['20'], ['512'], ['0.2', '0.4', '0.6'], 
                             ['None'], ['llmpruner-prune', 'llmpruner-tune'], ['asyncinter'], ['None'], ['None'],
                            ['None']]]
            CIFAR10_controls_9 = make_controls(script_name, init_seeds, device, resume_mode, control_name)
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
                run_time = '01:45:00'
            elif 'llama-2-13b' in controls[i][4] or 'opt-13b' in controls[i][4] or 'llama-3-8b' in controls[i][4]:
                run_time = '01:45:00'
            elif 'llama-2-70b' in controls[i][4]:
                run_time = '01:45:00'
            j += 1
            i += 1
            
        
        # print('isgpt', is_gpt)
        temp_mem = mem
        if is_llama:
            if '7b' in controls[i-1][4] or '6.7b' in controls[i-1][4] or '1.3b' in controls[i-1][4] or '8b' in controls[i-1][4]:
                temp_mem = int(2.5 * mem)
            elif '13b' in controls[i-1][4]:
                temp_mem = int(4.3 * mem)
            elif '30b' in controls[i-1][4]:
                temp_mem = int(7.5 * mem)
            elif '70b' in controls[i-1][4]:
                temp_mem = int(13.5 * mem)
            # temp_mem = int(3.5 * mem)
        if is_opt:
            temp_mem = int(1.5 * temp_mem)
        if is_gpt:
            temp_mem = int(1.5 * temp_mem)


        gpu_num = 1
        # if 'clm' in controls[i-1][4] and '2048' in controls[i-1][4]:
        #     if '30b' in controls[i-1][4] or '70b' in controls[i-1][4]:
        #     #     gpu_num = 3
        #     # else:
        #     #     gpu_num = 2
        #         gpu_num = 2
        #     else:
        #         gpu_num = 2
        if 'clm' in controls[i-1][4] and '1024' in controls[i-1][4]:
            if '70b' in controls[i-1][4]:
                gpu_num = 6
            elif '13b' in controls[i-1][4]:
                gpu_num = 2
            else:
                gpu_num = 1

            # if 'probe' in controls[i-1][4]:
            #     if '30b' in controls[i-1][4] or '70b' in controls[i-1][4]:
            #         gpu_num = 2
            #     else:
            #         gpu_num = 1
            # else:
            #     if '30b' in controls[i-1][4] or '70b' in controls[i-1][4]: 
            #         gpu_num = 3
            #     elif '13b' in controls[i-1][4]:
            #         gpu_num = 2
            #     else:
            #         gpu_num = 1
        if 'csr' in controls[i-1][4]:
            if '70b' in controls[i-1][4]:
                gpu_num = 6
            elif '13b' in controls[i-1][4]:
                gpu_num = 2
            else:
                gpu_num = 1

        s = '#!/bin/bash -l\n'
        
        s += f'#SBATCH --time={run_time}\n'
        s += f'#SBATCH --nodes={task_parallel_num}\n'
        s += f'#SBATCH --ntasks={task_parallel_num}\n'
        # s += '#SBATCH --cpus-per-task=2'
        # s += '#SBATCH --gres=gpu:a100:1\n'
       

        # s += f'#SBATCH -A aanwar\n'
        # s += f'#SBATCH --gres=gpu:a100:{gpu_num}\n'
        # s += '#SBATCH --partition=a100-4\n'

        s += f'#SBATCH -A dingj\n'
        s += f'#SBATCH --gres=gpu:{gpu_num}\n'
        s += '#SBATCH --partition=jd-4a100\n'


        s += f'#SBATCH --mem={temp_mem}gb\n'
        # s += '#SBATCH --mail-type=ALL \n'
        # s += '#SBATCH --mail-user=le000288@umn.edu\n'
        s += f'#SBATCH -o {res_path}/%j_{filename}.out\n'
        s += f'#SBATCH -e {res_path}/%j_{filename}.err\n'
        s += '\n'
        s += f'cd /home/aanwar/le000288/{code_folder}/src\n'
        s += '\n'
        s += 'export PATH=/home/aanwar/le000288/miniconda3/envs/pp/bin:$PATH\n'
        # if 'max' in controls[i][-1]:
        #     s_for_max = s_for_max + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
        #         '--resume_mode {} --log_interval {} --device {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])

        #     if k_for_max % round == round - 1:
        #         s_for_max = s_for_max[:-2] + '\nwait\n'
        #     k_for_max = k_for_max + 1
        #     continue
        # while i < len(controls):
        # srun --nodes=1 --ntasks=1 
        # srun --nodes=1 --ntasks=1 
        for item in sub_controls:
            s += '\n'
            s = s + 'python {} --device {} --resume_mode {} --init_seed {} --control_name {}\n'.format(*item)
        
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
        command = f'sbatch {filename}.pbs --wait\n'
        run_file.write(command)
        run_file.close()

        with open(bash_file_name, 'r') as cur_file:
            line_count = sum(1 for line in cur_file)

        if line_count > 180:
            bash_file_name = './{}.bash'.format(f'msi_{file}_{data[0]}_{i}')
            print('bash_file_name', bash_file_name)
            delete_file_if_exist(bash_file_name)
    return


if __name__ == '__main__':
    main()
