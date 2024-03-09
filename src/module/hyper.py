import torch
import re
from config import cfg

MULTIGPUS_MODEL_NAME_LIST = ['llama-2-70b']

def process_control():
    print('torch version: ', torch.__version__)
    print('cuda version: ', torch.version.cuda)
    print('cudnn version: ', torch.backends.cudnn.version())

    cfg['cudatoolkit_version'] = float(torch.version.cuda)
    cfg['cudnn_version'] = float(torch.backends.cudnn.version())
    cfg['gpu_type'] = 'A100'
    cfg['model_name'] = cfg['control']['model_name']
    cfg['task_name'] = cfg['control']['task_name']
    cfg['batch_size'] = int(cfg['control']['batch_size'])
    cfg['seq_len'] = int(cfg['control']['seq_len'])
    cfg['prune_hyper'] = float(cfg['control']['prune_hyper'])
    cfg['calibration_stage'] = False
    # no skip
    cfg['skip'] = 2
    
    prune_name_list = cfg['control']['prune_name'].split('+')

    cfg['prune_name'] = prune_name_list[0]
    cfg['prune_metric'] = None
    cfg['probe_num'] = 1
    prune_name_sub_list = cfg['prune_name'].split('-')
    if len(prune_name_sub_list) > 1:
        cfg['prune_method'] = prune_name_sub_list[1]
        cfg['prune_metric'] = prune_name_sub_list[2]
        if 'probe' in cfg['prune_method']:
            cfg['qk_proj_prune'] = prune_name_sub_list[3]  
            # fill or each
            cfg['vo_proj_prune'] = prune_name_sub_list[4]
        
        if 'svd' in cfg['prune_method']:
            match = re.search(r'svd(\d+\.\d+)', cfg['prune_metric'])
            if match:
                # Convert the matched string to a float
                float_value = float(match.group(1))
            else:
                float_value = None  # Or some default value or error handling
            cfg['svd_ratio'] = float_value

        # if 'skip' in cfg['prune_method']:
        #     match = re.search(r'skip(\d+)', cfg['prune_method'])
        #     if match:
        #         # Convert the matched string to a float
        #         int_value = int(match.group(1))
        #     cfg['skip'] = int_value

        if 'calib' in cfg['prune_method']:
            calib_info_list = prune_name_list[1].split('-')
            cfg['calibration_dataset'] = calib_info_list[0]
            cfg['calibration_nsamples'] = calib_info_list[1]
            # set all to all samples in the calibration dataset
            if cfg['calibration_nsamples'] != 'all':
                cfg['calibration_nsamples'] = int(cfg['calibration_nsamples'])  

        if 'probe' in cfg['prune_method'] and 'probefixratio' in cfg['prune_method']:
            match = re.search(r'probefixratio(\d+\.\d+)', cfg['prune_method'])
            if match:
                # Convert the matched string to a float
                float_value = float(match.group(1))
            else:
                float_value = None
            
            cfg['probefixratio'] = float_value   

        if 'multiprobe' in cfg['prune_method']:
            match = re.search(r'multiprobe(\d+)', cfg['prune_method'])
            if match:
                # Convert the matched string to a float
                int_value = int(match.group(1))
            else:
                int_value = None
            
            cfg['probe_num'] = int_value   
            # cfg['probe_size'] = 

        if 'async' in cfg['prune_method']:
            match = re.search(r'async(\d+\.\d+)', cfg['prune_method'])
            if match:
                # Convert the matched string to a float
                float_value = float(match.group(1))
            else:
                float_value = None
            
            cfg['asyncratio'] = float_value   

        if 'ema' in cfg['prune_method']:
            match = re.search(r'ema(\d+\.\d+)', cfg['prune_method'])
            if match:
                # Convert the matched string to a float
                float_value = float(match.group(1))
            else:
                float_value = None  # Or some default value or error handling
            cfg['ema_momentum'] = float_value
    else:
        cfg['prune_method'] = ''

    cfg['cust_tgt_modules'] = cfg['control']['cust_tgt_modules'].split('+')
    if 'llama' in cfg['model_name'] and cfg['cust_tgt_modules'] != ['default']:
        cfg['cust_tgt_modules'] = [module.replace('-', '_') for module in cfg['cust_tgt_modules']]
    elif cfg['cust_tgt_modules'] == ['default']:
        if 'fixprune' in cfg['prune_method']:
            cfg['cust_tgt_modules'] = TRANSFORMERS_MODELS_TO_EWI_TARGET_MODULES_MAPPING[cfg['model_name']]
        else:
            cfg['cust_tgt_modules'] = TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING[cfg['model_name']]

    cfg['prune_dim'] = -1
    cfg['pq_p'] = 1
    cfg['pq_q'] = 2
    cfg['pq_beta'] = 0.9
    cfg['pq_gamma'] = 1
    make_data_name()
    if cfg['task_name'] in ['s2s', 'sc', 'clm', 't2i', 'csr']:
        cfg['collate_mode'] = 'transformer'
        cfg['gpt2'] = {'max_length': 512}
        if 'llama' in cfg['model_name']:
            # reassign in make_hf_model
            cfg[cfg['model_name']] = {'max_length': None}
        if 'opt' in cfg['model_name']:
            # reassign in make_hf_model
            cfg[cfg['model_name']] = {'max_length': None}
        # cfg['opt'] = {'max_length': 128}
    elif cfg['task_name'] in ['ic']:
        cfg['collate_mode'] = 'dict'
        data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                      'CIFAR100': [3, 32, 32]}
        target_size = {'MNIST': 10, 'FashionMNIST': 10, 'SVHN': 10, 'CIFAR10': 10, 'CIFAR100': 100}
        cfg['linear'] = {}
        cfg['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
        cfg['cnn'] = {'hidden_size': [64, 128, 256, 512]}
        cfg['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
        cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
        cfg['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
        cfg['data_shape'] = data_shape[cfg['data_name']]
        cfg['target_size'] = target_size[cfg['data_name']]
    else:
        raise ValueError('Not valid task name')
    model_name = cfg['model_name']
    if model_name not in cfg:
        cfg[model_name] = {}
    cfg[model_name]['shuffle'] = {'train': False, 'test': False}
    if cfg['task_name'] in ['s2s', 'sc', 'clm', 'csr']:
        cfg[model_name]['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    elif cfg['task_name'] in ['ic']:
        cfg[model_name]['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    elif cfg['task_name'] in ['t2i']:
        cfg['collate_mode'] = 'dreambooth'
        cfg[model_name]['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    else:
        raise ValueError('Not valid task name')

    cfg['probe_size'] = int(cfg[model_name]['batch_size']['test'] / cfg['probe_num'])
    if cfg[model_name]['batch_size']['test'] % cfg['probe_num'] != 0:
        raise ValueError('probe_num needs to be divisible by batch size')
    
    cfg['logger_detailed_info'] = False
    print(cfg['prune_hyper'] == 9999)
    print('cfg: ', cfg)
    return


def make_data_name():
    data_name_list = cfg['control']['data_name'].split('-')
    print('data_name_list', data_name_list)
    if len(data_name_list) == 2:
        cfg['data_name'], cfg['subset_name'] = data_name_list
    else:
        cfg['data_name'] = data_name_list[0]
        cfg['subset_name'] = 'none'
    if cfg['data_name'] == 'arcchallenge':
        cfg['data_name'] = 'arc_challenge'
    elif cfg['data_name'] == 'arceasy':
        cfg['data_name'] = 'arc_easy'
    if cfg['python_file'] == 'test_dense_harness_model.py':
        return
    if cfg['task_name'] in ['s2s', 'sc', 'clm', 'csr', 't2i']:
        data_name_dict = {
            # https://huggingface.co/datasets/wikitext
            'wikitext': {'data_name': 'wikitext',
                          'subset_name_dict': {'2v1': {'subset_name': 'wikitext-2-raw-v1',
                                                   'text_column': ['text'],
                                                   'label_column': None}
                                           }                       
                         },
            # piqa: piqa
            # boolq: boolq , 
            # arc-e: arc-easy 
            # arc-c: arc-challenge (Clark et al., 2018), 
            # hellaswag: hellaswag (Zellers et al., 2019) 
            # winogrande: winogrande 
            # obqa: OpenBookQA (Mihaylov et al., 2018)
            # preprocessing according to: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/arc.py
            # https://huggingface.co/datasets/piqa
            'boolq': {'data_name': 'google/boolq',
                    'subset_name_dict': {
                        'none': {'subset_name': None,
                              'text_column': ['hardcode'],
                              'label_column': 'hardcode'}
                        },
            },  
            'piqa': {'data_name': 'piqa',
                    'subset_name_dict': {
                        'none': {'subset_name': None,
                              'text_column': ['hardcode'],
                              'label_column': 'hardcode'}
                        },
            },          
            # https://huggingface.co/datasets/social_i_qa/viewer/default/validation
            'siqa': {'data_name': 'social_i_qa',
                    'subset_name_dict': {
                        'none': {'subset_name': None,
                              'text_column': ['hardcode'],
                              'label_column': 'hardcode'}
                        },
            },         
            # https://huggingface.co/datasets/ai2_arc          
           'arc': {'data_name': 'ai2_arc',
                    'subset_name_dict': {
                        'e': {'subset_name': 'ARC-Easy',
                              'text_column': ['hardcode'],
                             'label_column': 'hardcode'},   
                        'c': {'subset_name': 'ARC-Challenge',
                              'text_column': ['hardcode'],
                              'label_column': 'hardcode'}
                        },                        
            },
            # https://huggingface.co/datasets/Rowan/hellaswag
            'hellaswag': {'data_name': 'Rowan/hellaswag',
                    'subset_name_dict': {
                        'none': {'subset_name': None,
                              'text_column': 'hardcode',
                              'label_column': 'hardcode'}, 
                        },                        
            },
            'winogrande': {'data_name': 'winogrande',
                    'subset_name_dict': {
                        'none': {'subset_name': 'winogrande_debiased',
                              'text_column': 'hardcode',
                              'label_column': 'hardcode'}, 
                        },                        
            },
            # https://huggingface.co/datasets/openbookqa
            'obqa': {'data_name': 'openbookqa',
                    'subset_name_dict': {
                        'main': {'subset_name': 'main',
                              'text_column': ['hardcode'],
                              'label_column': 'hardcode'},    
                        },                        
            },
            # Dataset: https://github.com/google/dreambooth
            # DreamBooth paper: https://arxiv.org/pdf/2208.12242.pdf
            'dreambooth': {'data_name': 'DreamBooth',
                           'subset_name_dict': {
                               'backpack': {'subset_name': 'backpack',
                                            'class': 'backpack',
                                            'category': 'object'},
                               'backpack_dog': {'subset_name': 'backpack_dog',
                                                'class': 'backpack',
                                                'category': 'object'},
                               'bear_plushie': {'subset_name': 'bear_plushie',
                                                'class': 'stuffed animal',
                                                'category': 'toy'},
                               'berry_bowl': {'subset_name': 'berry_bowl',
                                              'class': 'bowl',
                                              'category': 'object'},
                               'can': {'subset_name': 'can', 'class': 'can', 'category': 'object'},
                               'candle': {'subset_name': 'candle', 'class': 'candle', 'category': 'object'},
                               'cat': {'subset_name': 'cat', 'class': 'cat', 'category': 'live object'},
                               'cat2': {'subset_name': 'cat2', 'class': 'cat', 'category': 'live object'},
                               'clock': {'subset_name': 'clock', 'class': 'clock', 'category': 'object'},
                               'colorful_sneaker': {'subset_name': 'colorful_sneaker',
                                                    'class': 'sneaker',
                                                    'category': 'object'},
                               'dog': {'subset_name': 'dog', 'class': 'dog', 'category': 'live object'},
                               'dog2': {'subset_name': 'dog2', 'class': 'dog', 'category': 'live object'},
                               'dog3': {'subset_name': 'dog3', 'class': 'dog', 'category': 'live object'},
                               'dog5': {'subset_name': 'dog5', 'class': 'dog', 'category': 'live object'},
                               'dog6': {'subset_name': 'dog6', 'class': 'dog', 'category': 'live object'},
                               'dog7': {'subset_name': 'dog7', 'class': 'dog', 'category': 'live object'},
                               'dog8': {'subset_name': 'dog8', 'class': 'dog', 'category': 'live object'},
                               'duck_toy': {'subset_name': 'duck_toy', 'class': 'toy', 'category': 'toy'},
                               'fancy_boot': {'subset_name': 'fancy_boot',
                                              'class': 'boot',
                                              'category': 'object'},
                               'grey_sloth_plushie': {'subset_name': 'grey_sloth_plushie',
                                                      'class': 'stuffed animal',
                                                      'category': 'toy'},
                               'monster_toy': {'subset_name': 'monster_toy',
                                               'class': 'toy',
                                               'category': 'toy'},
                               'pink_sunglasses': {'subset_name': 'pink_sunglasses',
                                                   'class': 'glasses',
                                                   'category': 'accessory'},
                               'poop_emoji': {'subset_name': 'poop_emoji',
                                              'class': 'toy',
                                              'category': 'toy'},
                               'rc_car': {'subset_name': 'rc_car', 'class': 'toy', 'category': 'toy'},
                               'red_cartoon': {'subset_name': 'red_cartoon',
                                               'class': 'cartoon',
                                               'category': 'object'},
                               'robot_toy': {'subset_name': 'robot_toy', 'class': 'toy', 'category': 'toy'},
                               'shiny_sneaker': {'subset_name': 'shiny_sneaker',
                                                 'class': 'sneaker',
                                                 'category': 'object'},
                               'teapot': {'subset_name': 'teapot', 'class': 'teapot', 'category': 'object'},
                               'vase': {'subset_name': 'vase', 'class': 'vase', 'category': 'object'},
                               'wolf_plushie': {'subset_name': 'wolf_plushie',
                                                'class': 'stuffed animal',
                                                'category': 'toy'}}
                           }
        }
        if cfg['data_name'] == 'dreambooth':
            cfg['unique_id'] = 'sks'
            cfg['unique_class'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['class']
        else:
            cfg['hf_data_name'] = data_name_dict[cfg['data_name']]['data_name']
            cfg['hf_subset_name'] = data_name_dict[cfg['data_name']]['subset_name_dict'][
                cfg['subset_name']]['subset_name']
            cfg['text_column'] = data_name_dict[cfg['data_name']]['subset_name_dict'][
                cfg['subset_name']]['text_column']
            cfg['label_column'] = data_name_dict[cfg['data_name']]['subset_name_dict'][
                cfg['subset_name']]['label_column']
    return


TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama-2-7b": ["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
    # "llama": ["proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],

    'resnet9': ['.shortcut', '.conv1', '.conv2'],
    # 'resnet9': ['.conv2'],
    'resnet18': ['.shortcut', '.conv1', '.conv2'],

    'test': ['fc']
}

TRANSFORMERS_MODELS_TO_EWI_TARGET_MODULES_MAPPING = {
    "opt": ["out_proj", "fc2"],
    "llama": ["o_proj", "down_proj"],
    'resnet9': ['.shortcut', '.conv1', '.conv2'],
    # 'resnet9': ['.conv2'],
    'resnet18': ['.shortcut', '.conv1', '.conv2'],
    'test': ['fc']
}

TRANSFORMERS_MODELS_OUT_TARGET_MODULES_MAPPING = {
    "llama": ["gate_proj", "up_proj", "q_proj", "k_proj", "v_proj"],
    'opt': ["k_proj", "v_proj", "q_proj", "fc1"]
}


# gpt2 layer
'''
key:  transformer.h.3 <class 'transformers.models.gpt2.modeling_gpt2.GPT2Block'>
key:  transformer.h.3.ln_1 <class 'torch.nn.modules.normalization.LayerNorm'>
key:  transformer.h.3.attn <class 'transformers.models.gpt2.modeling_gpt2.GPT2Attention'>
key:  transformer.h.3.attn.c_attn <class 'transformers.pytorch_utils.Conv1D'>
key:  transformer.h.3.attn.c_proj <class 'transformers.pytorch_utils.Conv1D'>
key:  transformer.h.3.attn.attn_dropout <class 'torch.nn.modules.dropout.Dropout'>
key:  transformer.h.3.attn.resid_dropout <class 'torch.nn.modules.dropout.Dropout'>
key:  transformer.h.3.ln_2 <class 'torch.nn.modules.normalization.LayerNorm'>
key:  transformer.h.3.mlp <class 'transformers.models.gpt2.modeling_gpt2.GPT2MLP'>
key:  transformer.h.3.mlp.c_fc <class 'transformers.pytorch_utils.Conv1D'>
key:  transformer.h.3.mlp.c_proj <class 'transformers.pytorch_utils.Conv1D'>
key:  transformer.h.3.mlp.act <class 'transformers.activations.NewGELUActivation'>
key:  transformer.h.3.mlp.dropout <class 'torch.nn.modules.dropout.Dropout'>
'''


# opt 1.3b layer

'''
125M、350M、1.3B、2.7B、6.7B、13B、30B、66B、175B
selected: 6.7B、13B、30B、66B
key:  model.decoder.layers.0 <class 'transformers.models.opt.modeling_opt.OPTDecoderLayer'>
key:  model.decoder.layers.0.self_attn <class 'transformers.models.opt.modeling_opt.OPTAttention'>
key:  model.decoder.layers.0.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
key:  model.decoder.layers.0.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
key:  model.decoder.layers.0.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
key:  model.decoder.layers.0.self_attn.out_proj <class 'torch.nn.modules.linear.Linear'>
key:  model.decoder.layers.0.activation_fn <class 'torch.nn.modules.activation.ReLU'>
key:  model.decoder.layers.0.self_attn_layer_norm <class 'torch.nn.modules.normalization.LayerNorm'>
key:  model.decoder.layers.0.fc1 <class 'torch.nn.modules.linear.Linear'>
key:  model.decoder.layers.0.fc2 <class 'torch.nn.modules.linear.Linear'>
key:  model.decoder.layers.0.final_layer_norm <class 'torch.nn.modules.normalization.LayerNorm'>
'''

# llama-2-7b layer
'''
7b, 13b, 65b
key:  model.layers.0 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
key:  model.layers.0.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
key:  model.layers.0.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
key:  model.layers.0.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
key:  model.layers.0.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
key:  model.layers.0.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
key:  model.layers.0.self_attn.rotary_emb <class 'transformers.models.llama.modeling_llama.LlamaRotaryEmbedding'>
key:  model.layers.0.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
key:  model.layers.0.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
key:  model.layers.0.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
key:  model.layers.0.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
key:  model.layers.0.mlp.act_fn <class 'transformers.activations.SiLUActivation'>
key:  model.layers.0.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
key:  model.layers.0.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
'''
