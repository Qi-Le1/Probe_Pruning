import os
import re
import math
import torch
import torch.nn as nn
from config import cfg
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, \
    AutoTokenizer, LlamaTokenizer, AutoModelForMultipleChoice, AutoModel, AutoConfig
# from transformers import LlamaForCausalLM
from .hf.modeling_llama import LlamaForCausalLM
from .hf.modeling_opt import OPTForCausalLM
from module import MULTIGPUS_MODEL_NAME_LIST, TRANSFORMERS_MODELS_OUT_TARGET_MODULES_MAPPING, alternate_broadcast
from accelerate import infer_auto_device_map ,init_empty_weights

# def check_multiple_for_tensor_cores(data_type):
#     if data_type == torch.float16:
#         if cfg['cudatoolkit_version'] >= 11 and cfg['cudnn_version'] >= 7630:
#             if cfg['gpu_type'] == 'A100':
#                 return 64
#             else:
#                 return 8


def make_hf_model(model_name, sub_model_name=None):


    if model_name in MULTIGPUS_MODEL_NAME_LIST:
        device_map = "auto"
        # low_cpu_mem_usage = True
    else:
        # device_map = cfg['device']
        # if 'llama' in model_name:
        #     match = re.search(r'(\d+)b', model_name)
        #     number_before_b = match.group(1)
        #     approximate_gpu_memory_gb = int(number_before_b) * 2 + cfg['batch_size'] * cfg['seq_len'] * 11000 * 8 / 1024 / 1024 / 1024 + 5
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
    print('device_map', device_map)
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
    elif 'llama-2' in model_name:
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
            config = AutoConfig.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['model_name_or_path'])
            num_hidden_layers = config.num_hidden_layers
            if 'gridsearch' not in cfg['prune_method']:
                cfg['prune_ratio'] = num_hidden_layers / (num_hidden_layers - (cfg['skip_layers'] + 1)) * cfg['prune_ratio'] 
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
    
    if any(k in cfg['model_name_or_path'] for k in ("gpt", "opt", "bloom", "llama")):
        padding_side = "left"
        # produce nan if we pad input text to the left
        # if cfg['task_name'] == 'csr':
        #     padding_side = "right"
    else:
        padding_side = "right"

    if any(k in cfg['model_name_or_path'] for k in ("opt", "llama")):
        if cfg['seq_len'] > model.config.max_position_embeddings:
            raise ValueError(
                f"seq_len ({cfg['seq_len']}) is larger than max_position_embeddings ({model.config.max_position_embeddings})."
            )
        cfg[cfg['model_name']]['max_length'] = cfg['seq_len'] 
        # cfg[cfg['model_name']]['max_length'] = 128
        # cfg[cfg['model_name']]['max_length'] = 512
        print('max_length', cfg[cfg['model_name']]['max_length'])
    if 'llama' in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_tokenizer_path'],
                                                   padding_side=padding_side)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_name_or_path'], cache_dir=cfg['cache_tokenizer_path'],
                                                  padding_side=padding_side)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if any(k in model_name for k in ("gpt", "llama-2")):
        model.config.pad_token_id = tokenizer.pad_token_id
    if 'opt' in model_name:
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
    cfg['pad_token_id'] = tokenizer.pad_token_id    

    # if 'svd' in cfg['prune_method']:
    #     def low_rank_approximation(U, S, V, ratio):
    #         U, S, V = U.to(torch.float16), S.to(torch.float16), V.to(torch.float16)
    #         remain_count = int(hidden_size * ratio)
    #         k = math.ceil(remain_count / multiple) * multiple
    #         print('k', k)
    #         S_k = torch.diag(S[:k]).to(device_map)
    #         U_k = U[:, :k].to(device_map)
    #         V_k = V[:k, :].to(device_map)
    #         return U_k, S_k, V_k

    #     def cal_cumulative_contributions(S):
    #         total_variance = torch.sum(S.float() ** 2)
    #         # 计算每个奇异值的累积贡献率
    #         cumulative_contributions = torch.cumsum(S.float() ** 2, dim=0) / total_variance
    #         # 打印累积贡献率
    #         for i, contribution in enumerate(cumulative_contributions, start=1):
    #             if i == 600:
    #                 return
    #             print(f"Top {i} singular values' cumulative contribution: {contribution.item():.4f}")
    #         return

    #     if 'llama' in model_name:
    #         with torch.no_grad():
    #             model.train(False)
    #             hidden_size = model.config.hidden_size
    #             num_heads = model.config.num_attention_heads
    #             num_key_value_heads = model.config.num_key_value_heads
    #             head_dim = model.config.hidden_size // num_heads
    #             num_key_value_groups = num_heads // num_key_value_heads

    #             multiple = check_multiple_for_tensor_cores(torch.float16)
                
                
    #             layers = model.model.layers
    #             for layer in layers:               
    #                 print('\nlayer_order', layer.mlp.layer_order)
    #                 print('layer.mlp.gate_proj.device', layer.mlp.gate_proj.weight.data.device)
    #                 U, S, V = torch.linalg.svd(layer.mlp.gate_proj.weight.data.float(), full_matrices=False)
    #                 cal_cumulative_contributions(S)
    #                 U, S, V = low_rank_approximation(U, S, V, cfg['svd_ratio'])
    #                 layer.mlp.gate_proj_svd_U = nn.Parameter(U).to(device_map)
    #                 layer.mlp.gate_proj_svd_S = nn.Parameter(S).to(device_map)
    #                 layer.mlp.gate_proj_svd_V = nn.Parameter(V).to(device_map)
    #                 layer.mlp.gate_proj_svd_U.requires_grad = False
    #                 layer.mlp.gate_proj_svd_S.requires_grad = False
    #                 layer.mlp.gate_proj_svd_V.requires_grad = False
    #                 quantiles_to_calculate = [0.25, 0.5, 0.75, 0.9, 0.95, 1]  # For example, the first, second (median), and third quartiles
    #                 for q in quantiles_to_calculate:
    #                     quantile_value = torch.quantile(S.float(), q)
    #                     print(f"Quantile {q}: {quantile_value}")

                    

    #                 print('layer.mlp.up_proj.device', layer.mlp.up_proj.weight.data.device)
    #                 U, S, V = torch.linalg.svd(layer.mlp.up_proj.weight.data.float(), full_matrices=False)
    #                 cal_cumulative_contributions(S)
    #                 U, S, V = low_rank_approximation(U, S, V, cfg['svd_ratio'])
    #                 layer.mlp.up_proj_svd_U = nn.Parameter(U).to(device_map)
    #                 layer.mlp.up_proj_svd_S = nn.Parameter(S).to(device_map)
    #                 layer.mlp.up_proj_svd_V = nn.Parameter(V).to(device_map)
    #                 layer.mlp.up_proj_svd_U.requires_grad = False
    #                 layer.mlp.up_proj_svd_S.requires_grad = False
    #                 layer.mlp.up_proj_svd_V.requires_grad = False
    #                 quantiles_to_calculate = [0.25, 0.5, 0.75, 0.9, 0.95, 1]  # For example, the first, second (median), and third quartiles
    #                 for q in quantiles_to_calculate:
    #                     quantile_value = torch.quantile(S.float(), q)
    #                     print(f"Quantile {q}: {quantile_value}")
                    

    #                 print('layer.self_attn.q_proj.device', layer.self_attn.q_proj.weight.data.device)
    #                 U, S, V = torch.linalg.svd(layer.self_attn.q_proj.weight.data.float(), full_matrices=False)
    #                 cal_cumulative_contributions(S)
    #                 U, S, V = low_rank_approximation(U, S, V, cfg['svd_ratio'])
    #                 # layer.self_attn.q_proj_svd_U = nn.Parameter(U).to(device_map)
    #                 # layer.self_attn.q_proj_svd_S = nn.Parameter(S).to(device_map)
    #                 # layer.self_attn.q_proj_svd_V = nn.Parameter(V).to(device_map)
    #                 quantiles_to_calculate = [0.25, 0.5, 0.75, 0.9, 0.95, 1]  # For example, the first, second (median), and third quartiles
    #                 for q in quantiles_to_calculate:
    #                     quantile_value = torch.quantile(S.float(), q)
    #                     print(f"Quantile {q}: {quantile_value}")
                    

    #                 print('layer.self_attn.k_proj.device', layer.self_attn.k_proj.weight.data.device)
    #                 U, S, V = torch.linalg.svd(layer.self_attn.k_proj.weight.data.float(), full_matrices=False)
    #                 cal_cumulative_contributions(S)
    #                 U, S, V = low_rank_approximation(U, S, V, cfg['svd_ratio'])
                    
    #                 # layer.self_attn.k_proj_svd_U = nn.Parameter(U).to(device_map)
    #                 # layer.self_attn.k_proj_svd_S = nn.Parameter(S).to(device_map)
    #                 # layer.self_attn.k_proj_svd_V = nn.Parameter(V).to(device_map)
    #                 quantiles_to_calculate = [0.25, 0.5, 0.75, 0.9, 0.95, 1]  # For example, the first, second (median), and third quartiles
    #                 for q in quantiles_to_calculate:
    #                     quantile_value = torch.quantile(S.float(), q)
    #                     print(f"Quantile {q}: {quantile_value}")

    #                 print('layer.self_attn.v_proj.device', layer.self_attn.v_proj.weight.data.device)
    #                 U, S, V = torch.linalg.svd(layer.self_attn.v_proj.weight.data.float(), full_matrices=False)
    #                 cal_cumulative_contributions(S)
    #                 U, S, V = low_rank_approximation(U, S, V, cfg['svd_ratio'])
    #                 # layer.self_attn.v_proj_svd_U = nn.Parameter(U).to(device_map)
    #                 # layer.self_attn.v_proj_svd_S = nn.Parameter(S).to(device_map)
    #                 # layer.self_attn.v_proj_svd_V = nn.Parameter(V).to(device_map)
    #                 quantiles_to_calculate = [0.25, 0.5, 0.75, 0.9, 0.95, 1]  # For example, the first, second (median), and third quartiles
    #                 for q in quantiles_to_calculate:
    #                     quantile_value = torch.quantile(S.float(), q)
    #                     print(f"Quantile {q}: {quantile_value}")
    #     elif 'opt' in model_name:
    #         with torch.no_grad():
    #             model.train(False)
    #             for name, parameter in model.named_parameters():
    #                 print('name', name)
    #                 if 'fc' in name and 'weight' in name:
    #                 # print('layer.mlp.gate_proj.device', layer.mlp.gate_proj.weight.data.device)
    #                     U, S, V = torch.linalg.svd(parameter.float(), full_matrices=False)
    #                     cal_cumulative_contributions(S)

        # torch.cuda.empty_cache()
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
