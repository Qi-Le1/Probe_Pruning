import os
import re
import math
import torch
import torch.nn as nn
from config import cfg
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, \
    AutoTokenizer, LlamaTokenizer, AutoModelForMultipleChoice, AutoModel, AutoConfig
# from transformers import LlamaForCausalLM

from module import MULTIGPUS_MODEL_NAME_LIST, TRANSFORMERS_MODELS_OUT_TARGET_MODULES_MAPPING
from accelerate import infer_auto_device_map ,init_empty_weights
from LLMPruner import PeftModel

# def check_multiple_for_tensor_cores(data_type):
#     if data_type == torch.float16:
#         if cfg['cudatoolkit_version'] >= 11 and cfg['cudnn_version'] >= 7630:
#             if cfg['gpu_type'] == 'A100':
#                 return 64
#             else:
#                 return 8

def llmpruner_load(model_type: str = 'pruneLLM', ckpt: str = '', lora_ckpt: str = ''):
    if model_type == 'pruneLLM':
        pruned_dict = torch.load(ckpt, map_location='cpu')
        model = pruned_dict['tokenizer'], pruned_dict['model']
    elif model_type == 'tune_prune_LLM':
        pruned_dict = torch.load(ckpt, map_location='cpu')
        model = pruned_dict['tokenizer'], pruned_dict['model']
        model = PeftModel.from_pretrained(
            model,
            lora_ckpt,
            torch_dtype=torch.float16,
        )
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if device == "cuda":
        model.half()
        model = model.cuda()

    # # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2
    return model


def make_local_tuned_model(model_name):
    if 'llama' in model_name:
        print('model_name', model_name)
        if 'llmpruner' in cfg['prune_method']:
            model_path = f"output/llmpruner/{cfg['init_seed']}_llmpruner_{model_name}_{cfg['prune_ratio']}/pytorch_model.bin"
            lora_path = f"output/llmpruner/{cfg['init_seed']}_llmpruner_{model_name}_{cfg['prune_ratio']}"

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"Model file not found: {lora_path}")
            
            if 'llmpruner-prune' in cfg['prune_method']:
                model = llmpruner_load(model_type='pruneLLM', ckpt=model_path)
            elif 'llmpruner-tune' in cfg['prune_method']:
                model = llmpruner_load(model_type='tune_prune_LLM', ckpt=model_path, lora_ckpt=lora_path)
            else:
                raise NotImplementedError
        elif 'loraprune' in cfg['prune_method']:
            pass
    else:
        raise ValueError('Not valid model name')
    
    if any(k in cfg['model_name_or_path'] for k in ("opt", "llama")):
        padding_side = "left"
    else:
        padding_side = "right"

    if any(k in cfg['model_name_or_path'] for k in ("opt", "llama")):
        if cfg['max_seq_len'] > model.config.max_position_embeddings:
            raise ValueError(
                f"seq_len ({cfg['max_seq_len']}) is larger than max_position_embeddings ({model.config.max_position_embeddings})."
            )

    if 'llama' in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side=padding_side)
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if any(k in model_name for k in ("llama")):
        model.config.pad_token_id = tokenizer.pad_token_id

    cfg['pad_token_id'] = tokenizer.pad_token_id    

    model.config.use_cache = False
    return model, tokenizer


def make_hf_model(model_name):
    from .hf.modeling_llama import LlamaForCausalLM
    from .hf.modeling_opt import OPTForCausalLM

    if model_name in MULTIGPUS_MODEL_NAME_LIST:
        device_map = "auto"
        # low_cpu_mem_usage = True
    else:
        # device_map = cfg['device']
        # if 'llama' in model_name:
        #     match = re.search(r'(\d+)b', model_name)
        #     number_before_b = match.group(1)
        #     approximate_gpu_memory_gb = int(number_before_b) * 2 + cfg['batch_size'] * cfg['max_seq_len'] * 11000 * 8 / 1024 / 1024 / 1024 + 5
        #     print('approximate_gpu_memory_gb', approximate_gpu_memory_gb)
        #     if cfg['gpu_name'] == 'NVIDIA GeForce RTX 4090':
        #         if approximate_gpu_memory_gb > 24:
        #             device_map = "auto"
        #     elif cfg['gpu_name'] == 'NVIDIA A100-SXM4-40GB':
        #         # A100
        #         if approximate_gpu_memory_gb > 40:
        #             # pass
        #             device_map = "auto"
        device_map = "auto"
        # low_cpu_mem_usage = False
    if 'opt' in model_name:
        # https://huggingface.co/facebook/opt-1.3b
        # if '1.3b' in model_name:
        #     cfg['model_name_or_path'] = 'facebook/opt-1.3b'
        #     cfg['tokenizer_name_or_path'] = 'facebook/opt-1.3b'

        # cant load it from hf online, has config file issue
        if model_name == 'opt-6.7b':
            cfg['model_name_or_path'] = f"output/{cfg['model_name']}"
            cfg['tokenizer_name_or_path'] = f"output/{cfg['model_name']}"
        else:
            cfg['model_name_or_path'] = f"facebook/{cfg['model_name']}"
            cfg['tokenizer_name_or_path'] = f"facebook/{cfg['model_name']}"
    elif 'llama' in model_name:
        # https://huggingface.co/docs/transformers/main/model_doc/llama2
        # FOLLOW the instruction to run the script: python convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir output/llama-2-7b
        # in the above py file, change line 270 to model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True), need float16 not bfloat16
        # support ["llama-2-7b"]
        # need tokenizer.model, tokenizer_config.json from https://huggingface.co/meta-llama/Llama-2-13b-hf/tree/main   (corresponding model type)
        cfg['model_name_or_path'] = f'output/{model_name}'
        cfg['tokenizer_name_or_path'] = f'output/{model_name}'
    else:
        raise ValueError('Not valid model name')
    cfg['cache_model_path'] = os.path.join('output', 'model', model_name)
    cfg['cache_tokenizer_path'] = os.path.join('output', 'tokenizer', model_name)
    print("cfg['model_name_or_path']", cfg['model_name_or_path'])
    if cfg['task_name'] == 'clm':
        if 'llama' in model_name:
            model = LlamaForCausalLM.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['model_name_or_path'],  torch_dtype=torch.float16, device_map=device_map)
            # model = LlamaForCausalLM.from_pretrained(cfg['model_name_or_path'], torch_dtype=torch.float16,
            #                                         device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage)
        elif 'opt' in model_name:
            # Load the model configuration
            model = OPTForCausalLM.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_model_path'], torch_dtype=torch.float16, device_map=device_map)
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_model_path'],
                                                         device_map=device_map)
    elif cfg['task_name'] == 'csr':  # Assuming 'csr' stands for common sense reasoning
        if 'llama' in model_name:
            # "Training Llama in float16 is not recommended and known to produce nan, as such the model should be trained in bfloat16.""
            model = LlamaForCausalLM.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['model_name_or_path'],  torch_dtype=torch.float16, device_map=device_map)
            # to fit flap and simplify for flops comparision
        elif 'opt' in model_name:
            model = OPTForCausalLM.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['model_name_or_path'], torch_dtype=torch.float16, device_map=device_map)
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_model_path'],
                                                    device_map=device_map)
    else:
        raise ValueError('Not valid task name')
    
    if any(k in cfg['model_name_or_path'] for k in ("opt", "llama")):
        padding_side = "left"
        # produce nan if we pad input text to the left
        # if cfg['task_name'] == 'csr':
        #     padding_side = "right"
    else:
        padding_side = "right"

    if any(k in cfg['model_name_or_path'] for k in ("opt", "llama")):
        if cfg['max_seq_len'] > model.config.max_position_embeddings:
            raise ValueError(
                f"seq_len ({cfg['max_seq_len']}) is larger than max_position_embeddings ({model.config.max_position_embeddings})."
            )
    if 'llama' in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_tokenizer_path'],
                                                   padding_side=padding_side)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_name_or_path'], cache_dir=cfg['cache_tokenizer_path'],
                                                  padding_side=padding_side)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if any(k in model_name for k in ("llama")):
        model.config.pad_token_id = tokenizer.pad_token_id
    if 'opt' in model_name:
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
    cfg['pad_token_id'] = tokenizer.pad_token_id    

    # to fit flap and simplify for flops comparision
    if 'llama-2-70b' in cfg['model_name']:
        with torch.no_grad():
            model.train(False)
            hidden_size = model.config.hidden_size
            num_heads = model.config.num_attention_heads
            num_key_value_heads = model.config.num_key_value_heads
            head_dim = model.config.hidden_size // num_heads
            num_key_value_groups = num_heads // num_key_value_heads

            layers = model.model.layers
            for layer in layers:                        
                if layer.self_attn.k_proj.weight.data.size(0) == num_key_value_heads * head_dim:
                    temp_weight_data = layer.self_attn.k_proj.weight.data.repeat_interleave(num_key_value_groups, dim=0, output_size=num_heads * head_dim)
                    temp_weight_data = temp_weight_data.type(layer.self_attn.k_proj.weight.data.dtype)
                    layer.self_attn.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=model.config.attention_bias)
                    layer.self_attn.k_proj.weight = nn.Parameter(temp_weight_data)
                    layer.self_attn.k_proj.cal_total_flops = True
                if layer.self_attn.v_proj.weight.data.size(0) == num_key_value_heads * head_dim:
                    temp_weight_data = layer.self_attn.v_proj.weight.data.repeat_interleave(num_key_value_groups, dim=0, output_size=num_heads * head_dim)
                    temp_weight_data = temp_weight_data.type(layer.self_attn.v_proj.weight.data.dtype)
                    layer.self_attn.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=model.config.attention_bias)
                    layer.self_attn.v_proj.weight = nn.Parameter(temp_weight_data)
                    layer.self_attn.v_proj.cal_total_flops = True
                    
                del temp_weight_data
                torch.cuda.empty_cache()
                layer.self_attn.num_key_value_heads = num_heads
                layer.self_attn.num_key_value_groups = 1

    model.config.use_cache = False
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
