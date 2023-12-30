import os
import torch
import torch.nn as nn
from config import cfg
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, \
    AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForMultipleChoice, AutoModel
from module import MULTIGPUS_MODEL_NAME_LIST


def make_hf_model(model_name, sub_model_name=None):
    if model_name in MULTIGPUS_MODEL_NAME_LIST:
        device_map = "auto"
        low_cpu_mem_usage = True
    else:
        device_map = cfg['device']
        low_cpu_mem_usage = False

    if 'bart' in model_name:
        cfg['model_name_or_path'] = 'facebook/{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = 'facebook/{}'.format(model_name)
    elif 'bloom' in model_name:
        cfg['model_name_or_path'] = 'bigscience/{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = 'bigscience/{}'.format(model_name)
    elif 'bart' in model_name:
        cfg['model_name_or_path'] = 'facebook/{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = 'facebook/{}'.format(model_name)
    elif 'roberta' in model_name:
        cfg['model_name_or_path'] = '{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = '{}'.format(model_name)
    elif 'gpt' in model_name:
        cfg['model_name_or_path'] = '{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = '{}'.format(model_name)
    elif 't5' in model_name:
        cfg['model_name_or_path'] = '{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = '{}'.format(model_name)
    elif 'sdiffusion' in model_name:
        cfg['model_name_or_path'] = 'CompVis/stable-diffusion-v1-4'
        cfg['tokenizer_name_or_path'] = 'CompVis/stable-diffusion-v1-4'
    elif 'open-llama' in model_name:
        # https://huggingface.co/openlm-research/open_llama_3b_v2
        # support ["open-llama-3b", "open-llama-7b"]
        if '3b' in model_name:
            cfg['model_name_or_path'] = 'openlm-research/open_llama_3b_v2'
            cfg['tokenizer_name_or_path'] = 'openlm-research/open_llama_3b_v2'
        elif '7b' in model_name:
            cfg['model_name_or_path'] = 'openlm-research/open_llama_7b_v2'
            cfg['tokenizer_name_or_path'] = 'openlm-research/open_llama_7b_v2'
    elif 'opt' in model_name:
        # https://huggingface.co/facebook/opt-1.3b
        if '1.3b' in model_name:
            cfg['model_name_or_path'] = 'facebook/opt-1.3b'
            cfg['tokenizer_name_or_path'] = 'facebook/opt-1.3b'
    elif 'llama-2-7b' in model_name:
        # https://huggingface.co/docs/transformers/main/model_doc/llama2
        # FOLLOW the instruction to run the script: python convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir output/llama-2-7b
        # support ["llama-2-7b"]
        cfg['model_name_or_path'] = 'output/llama-2-7b'
        cfg['tokenizer_name_or_path'] = 'output/llama-2-7b'
    else:
        raise ValueError('Not valid model name')
    cfg['cache_model_path'] = os.path.join('output', 'model', model_name)
    cfg['cache_tokenizer_path'] = os.path.join('output', 'tokenizer', model_name)
    if cfg['task_name'] == 'clm':
        if 'llama' in model_name:
            model = LlamaForCausalLM.from_pretrained(cfg['model_name_or_path'], torch_dtype=torch.float16,
                                                    device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage)
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg['model_name_or_path'], torch_dtype=torch.float16, cache_dir=cfg['cache_model_path'],
                                                         device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage)
    elif cfg['task_name'] == 's2s':
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_model_path'])
    elif cfg['task_name'] == 'sc':
        if cfg['subset_name'] in ['mnli']:
            model = AutoModelForSequenceClassification.from_pretrained(cfg['model_name_or_path'],
                                                                       cache_dir=cfg['cache_model_path'],
                                                                       num_labels=3)  # "num_labels" is set up in model.config
        elif cfg['subset_name'] in ['stsb']:
            model = AutoModelForSequenceClassification.from_pretrained(cfg['model_name_or_path'],
                                                                       cache_dir=cfg['cache_model_path'], num_labels=1)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(cfg['model_name_or_path'],
                                                                       cache_dir=cfg['cache_model_path'])
    if cfg['task_name'] == 'mc':  # Assuming 'mc' stands for commonsense reasoning
        if 'llama' in model_name:
            # "Training Llama in float16 is not recommended and known to produce nan, as such the model should be trained in bfloat16.""
            model = LlamaForCausalLM.from_pretrained(cfg['model_name_or_path'], torch_dtype=torch.float16,
                                                     device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage)
            # cache_dir=cfg['cache_model_path']
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg['model_name_or_path'], torch_dtype=torch.float16, cache_dir=cfg['cache_model_path'],
                                                    device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage)
    elif cfg['task_name'] == 't2i':
        if sub_model_name is None:
            model = DiffusionPipeline.from_pretrained(cfg['model_name_or_path'], safety_checker=None,
                                                      cache_dir=cfg['cache_model_path'])
        elif sub_model_name == 'vae':
            model = AutoencoderKL.from_pretrained(cfg['model_name_or_path'], subfolder="vae")
        elif sub_model_name == 'unet':
            model = UNet2DConditionModel.from_pretrained(cfg['model_name_or_path'], subfolder="unet")
        elif sub_model_name == 'text_encoder':
            text_encoder_cls = import_model_class_from_model_name_or_path(cfg['model_name_or_path'])
            model = text_encoder_cls.from_pretrained(
                cfg['model_name_or_path'], subfolder="text_encoder"
            )
    else:
        raise ValueError('Not valid task name')
    if any(k in cfg['model_name_or_path'] for k in ("gpt", "opt", "bloom", "llama")):
        
        padding_side = "left"
    else:
        padding_side = "right"

    if any(k in cfg['model_name_or_path'] for k in ("opt", "llama")):
        cfg[cfg['model_name']]['max_length'] = model.config.max_position_embeddings
        # cfg[cfg['model_name']]['max_length'] = 512
        print('max_length', cfg[cfg['model_name']]['max_length'])
    if 'llama' in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_tokenizer_path'],
                                                   padding_side=padding_side)
    elif 'sdiffusion' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_name_or_path'], subfolder="tokenizer",
                                                  cache_dir=cfg['cache_tokenizer_path'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_name_or_path'], cache_dir=cfg['cache_tokenizer_path'],
                                                  padding_side=padding_side)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if any(k in model_name for k in ("gpt", "llama")):
        model.config.pad_token_id = tokenizer.pad_token_id
    if 'opt' in model_name:
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
    cfg['pad_token_id'] = tokenizer.pad_token_id    
    return model, tokenizer


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
