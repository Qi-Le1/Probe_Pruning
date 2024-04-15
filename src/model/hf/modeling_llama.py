# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import time
import warnings
from typing import List, Optional, Tuple, Union

import copy
import torch
import traceback 
import threading
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers import LlamaConfig

from config import cfg
from ..pruning_module import HiddenRepresentationPruning, cal_intersection_ratio, cal_prune_metric, cal_calib_prune_metric, cal_attn_weight_prune_metric
from module import nearest_even_number
from torch.nn.functional import cosine_similarity
from .utils import generate_probe, cal_res_hidden_state_diff  
'''
Note: transformers 4.35.0 version
'''

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# def median_process(x, probe_num, probe_size):
#     # Apply absolute value to x
#     abs_x = torch.abs(x)
#     # Adjust the view to organize the data by probe_num and probe_size
#     reorganized_abs_x = abs_x.view(probe_num, probe_size, x.size(-2), x.size(-1))
#     # Use torch.median to get the median value across the probe_size dimension
#     median_across_bsz = reorganized_abs_x.median(dim=1, keepdim=False).values
#     return median_across_bsz

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils.AttentionMaskConverter._prepare_4d_attention_mask"
    )
    return AttentionMaskConverter._prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        print(self.weight.device)
        print(hidden_states.device)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, num_heads, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # print('self_dim', self.dim, inv_freq, inv_freq.shape, flush=True)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        # if 'each' in cfg['prune_name'] and ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']):
        #     # Expand embeddings for each head if we prune different indices for each head
        #     cos_emb = emb.cos().to(dtype).unsqueeze(0).repeat(self.num_heads, 1, 1)
        #     sin_emb = emb.sin().to(dtype).unsqueeze(0).repeat(self.num_heads, 1, 1)

        #     self.register_buffer("cos_cached", cos_emb, persistent=False)
        #     self.register_buffer("sin_cached", sin_emb, persistent=False)
        # else:
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor


        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # print('apply_rotary_pos_emb_cos', cos.shape, flush=True)
    # print('apply_rotary_pos_emb_sin', sin.shape, flush=True)

    # a = q*cos
    # b = rotate_half(q) * sin
    # print('apply_rotary_pos_emb_a', a.shape, flush=True)
    # print('apply_rotary_pos_emb_b', b.shape, flush=True)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def apply_rotary_pos_emb_for_prune_each_head(q, k, cos, sin, position_ids, probe_qk_out_dim_indices_for_rope, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    bsz = q.shape[0]
    num_heads = q.shape[1]
    seq_len = q.shape[2]
    head_dim = q.shape[3]
    

    cos = cos[position_ids].unsqueeze(unsqueeze_dim).repeat(bsz, num_heads, 1, 1)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim).repeat(bsz, num_heads, 1, 1)
    index_tensor = probe_qk_out_dim_indices_for_rope.unsqueeze(0).unsqueeze(2).expand(bsz, -1, seq_len, -1)

    # Use torch.gather to extract the elements for each head (positions are different for each head)
    cos = torch.gather(cos, -1, index_tensor)
    sin = torch.gather(sin, -1, index_tensor)
    # print('apply_rotary_pos_emb_for_prune_each_head_emb_cos2', cos.shape, flush=True)
    # a = q*cos
    # b = rotate_half(q) * sin
    # print('apply_rotary_pos_emb_for_prune_each_head_emb_a', a.shape, flush=True)
    # print('apply_rotary_pos_emb_for_prune_each_head_b', b.shape, flush=True)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed





class LlamaMLP(nn.Module):
    def __init__(self, config, layer_order):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # self.gate_proj.cal_total_flops = True
        # self.up_proj.cal_total_flops = True
        # self.down_proj.cal_total_flops = True

        self.act_fn = ACT2FN[config.hidden_act]

        self.layer_order = layer_order
        self.custom_duration = 0
        # self.cal_total_flops = True
        self.pruning_module = HiddenRepresentationPruning(cfg, f'llama_mlp_{layer_order}')
        self.running_mean = None

        self.probe_out_dim_indices = None

        self.input_norm_gate_weight = None
        self.input_norm_up_weight = None

        self.intersected_prune_indices = None
        # self.last_batch_probe_out = torch.zeros((cfg['probe_num'], cfg['seq_len'], self.intermediate_size), device=self.gate_proj.weight.device)
        self.last_batch_probe_out = None

        self.cur_batch = 0
        self.prev = None
    
    # kwargs['post_attn_residual'],
    def probe_process(self, x):
        # 1. generate probe
        # 2. run matrix multiplication
        # 3. calculate score
        # 4. extract metric

        # generate probe
        # rank / mean / absnml
        if cfg['gate_probe_ratio'] == cfg['up_probe_ratio']:
            probe, selected_indices = generate_probe(x, cfg['gate_probe_ratio'])
            print('xshape', x.shape, flush=True)
            print('linearprobeshape', probe.shape, flush=True)
        else:
            raise ValueError('gate_probe_num should be equal to up_probe_num for now')
        
        print('probe', probe.shape, flush=True)
        # run matrix multiplication
        if 'gate_proj' in cfg['cust_tgt_modules']:
            gate_out = self.act_fn(self.gate_proj(probe, cal_mlp_probe_out_dim_metric=True, selected_indices=selected_indices))
        else:
            gate_out = self.act_fn(self.gate_proj(x))
        
        if 'up_proj' in cfg['cust_tgt_modules']:
            up_out = self.up_proj(probe, cal_mlp_probe_out_dim_metric=True, selected_indices=selected_indices)
        else:
            up_out = self.up_proj(x)

        probe_out = gate_out * up_out
        # probe_out = torch.randn(1, cfg['seq_len'], 11008, dtype=torch.float16, device=probe_gate.device)

        # calculate score
        if 'calib' in cfg['prune_method'] or 'runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']:
            # TODO if 'saveseqdim' in cfg['prune_method']:
            #     probe_out_dim_metric, comined_probe_out = cal_prune_metric(probe_out, self.down_proj.weight.data, cfg['prune_metric'], global_input_distribution=self.down_proj.get_global_input_distribution()[0])
            # else:
            print('key', self.down_proj.key)
            probe_out_dim_metric, comined_probe_out = cal_prune_metric(probe_out, self.down_proj.weight.data, cfg['prune_metric'], global_metric_score_distribution=self.down_proj.get_global_metric_score_distribution(), selected_indices=selected_indices)
        else:
            probe_out_dim_metric, comined_probe_out = cal_prune_metric(probe_out, self.down_proj.weight.data, cfg['prune_metric'])

        if 'globalratio' in cfg['prune_method']:
            probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.down_proj.pruning_ratio)
        else:
            probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, cfg['tc_multiple'])

        # extract matrix
        self.gate_proj.prepare_async_weight(out_dim_indices=probe_out_dim_indices)
        self.up_proj.prepare_async_weight(out_dim_indices=probe_out_dim_indices)
        self.down_proj.prepare_async_weight(in_dim_indices=probe_out_dim_indices)
        print('extract weight', probe_out_dim_indices.shape, flush=True)
        return probe_out_dim_indices, probe_out
    
    
    def forward(self, x, **kwargs):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            bsz, _, _ = x.shape
            if ('down_proj' in cfg['cust_tgt_modules'] or 'up_proj' in cfg['cust_tgt_modules'] or 'gate_proj' in cfg['cust_tgt_modules']) and self.layer_order > cfg['skip_layers']:
                if cfg['calibration_stage'] == True:
                    # if 'calib' in cfg['prune_method'] :
                        # bsz, _, _ = x.shape
                        # time_start = time.time()
                        # probe_out_dim_indices = torch.arange(self.intermediate_size, dtype=torch.long).to(device=x.device)
                        # temp = self.act_fn(self.gate_proj(x, probe_out_dim_indices=probe_out_dim_indices)) * self.up_proj(x, probe_out_dim_indices=probe_out_dim_indices)
                        # kwargs['probe_in_dim_indices'] = probe_out_dim_indices
                        # down_proj = self.down_proj(temp, **kwargs)
                        # custom_duration = time.time() - time_start
                        # return down_proj
                    torch.cuda.synchronize(cfg['cuda_default_stream'])
                    mlp_duration_start = time.time()
                    time_start = time.time()
                    temp_gate = self.act_fn(self.gate_proj(x))
                    custom_duration = time.time() - time_start
                    print('custom_duration gate', custom_duration, flush=True)

                    time_start = time.time()
                    temp_up = self.up_proj(x)
                    custom_duration = time.time() - time_start
                    print('custom_duration up', custom_duration, flush=True)
                    time_start = time.time()
                    # print('original gateup', temp_gate * temp_up, flush=True)
                    down_proj = self.down_proj(temp_gate * temp_up)
                    custom_duration = time.time() - time_start
                    print('custom_duration down', custom_duration, flush=True)
                    torch.cuda.synchronize(cfg['cuda_default_stream'])
                    mlp_duration = time.time() - mlp_duration_start
                    print('mlp_duration', mlp_duration, flush=True)
                    del temp_gate, temp_up
                    return down_proj
                    # and self.layer_order >= 5
                # elif cfg['calibration_stage'] == False and self.layer_order >= 5:
                elif cfg['calibration_stage'] == False:
                    # print('zzz', self.layer_order, flush=True)
                    if 'probe' in cfg['prune_method']:
                        time_start = time.time()
                        # print('xshape', x.shape, flush=True)
                        

                        if cfg['mode'] == 'sync':
                            probe_out_dim_indices, probe_out = self.probe_process(x)
                            if cfg['onlyprobe'] == True:
                                # match the shape, and will not count the flops for this part
                                # down_proj = self.down_proj(probe_out, cal_mlp_probe_out_dim_metric=True)
                                # if down_proj.shape != (cfg['batch_size'], cfg['seq_len'], self.hidden_size):
                                down_proj = torch.zeros((cfg['batch_size'], cfg['seq_len'], self.hidden_size), device=x.device, dtype=x.dtype)
                                return down_proj
                        elif cfg['mode'] == 'asyncintra':
                            # if not, do full inference
                            if 'post_layernorm_attn_residual' in kwargs:
                                # print('post_layernorm_attn_residual', flush=True)
                                _, _ = self.probe_process(kwargs['post_layernorm_attn_residual'])
                                return
                        # if 'asyncfullinf' in cfg['prune_method']:
                        # print('probe_out gateup', probe_out, flush=True)

                        # incorporate the global distribution
                        # if 'intersect' in cfg['prune_method']:
                        # record intersection ratio
                        # full_gate_out = self.act_fn(self.gate_proj(x))
                        # full_up_out = self.up_proj(x)
                        # self.fullinf_vs_optimal_select_mean_intersection_ratio, self.probe_vs_optimal_select_mean_intersection_ratio, self.probe_vs_fullinf_select_mean_intersection_ratio, \
                        # self.fullinf_vs_optimal_prune_mean_intersection_ratio, self.probe_vs_optimal_prune_mean_intersection_ratio, self.probe_vs_fullinf_prune_mean_intersection_ratio = \
                        #     cal_intersection_ratio(full_gate_out * full_up_out, probe_out, self.down_proj.weight.data, self.pruning_module, multiple)
                        # if self.intersected_prune_indices is not None:
                        #     probe_out[..., self.intersected_prune_indices] = 0

                        # if 'async' in cfg['prune_method'] and 'savemetricseq' in cfg['prune_method']:
                        #     # temp = probe_out
                        #     # probe_out = self.last_batch_probe_out
                        #     # self.last_batch_probe_out = temp

                        #     if 'squareasync' in cfg['prune_method']:
                        #         if self.last_batch_probe_out is None:
                        #             norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2, min=cfg['data_type_min'], max=cfg['data_type_max']) / cfg['probe_num']
                        #             self.last_batch_probe_out = norm_probe_out_square.detach()
                        #             probe_out = torch.zeros(1, probe_out.size(-2), probe_out.size(-1), device=probe_out.device, dtype=probe_out.dtype)
                        #         else:
                        #         # self.last_batch_probe_out = self.last_batch_probe_out.to(probe_out.device)    
                        #             # if 'squareasyncabs' in cfg['prune_method']:
                        #             #     probe_out = torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2
                        #             #     abs_probe = torch.abs(probe_out).to(torch.float32)
                        #             #     abs_last_batch_probe = torch.abs(self.last_batch_probe_out).to(torch.float32)
                        #             #     sum_across_two_terms = abs_probe + abs_last_batch_probe
                        #             #     # proportion = abs_x / torch.sum(abs_x, dim=0, keepdim=True)
                        #             #     proportion = (abs_probe / (sum_across_two_terms + 1e-10)).to(abs_probe.dtype)
                        #             #     combined_probe = (probe_out * proportion + self.last_batch_probe_out * (1 - proportion)).to(probe_out.dtype)
                        #             #     probe_out, self.last_batch_probe_out = torch.sqrt(self.last_batch_probe_out).unsqueeze(0), combined_probe
                        #             # if 'squareasync' in cfg['prune_method']:
                        #             proportion = cfg['asyncratio']
                        #             probe_out = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2, min=cfg['data_type_min'], max=cfg['data_type_max']) / cfg['probe_num']
                        #             combined_probe = (self.last_batch_probe_out * proportion + probe_out * (1 - proportion)).to(probe_out.dtype)
                        #             # combined_probe = probe_out
                        #             probe_out, self.last_batch_probe_out = torch.sqrt(self.last_batch_probe_out).unsqueeze(0), combined_probe
                        #             # proportion = 10
                        #             # print('proportion ', proportion, flush=True)
                        #             # comp_across_bsz = torch.sum(x * proportion, dim=0)
                        #             # comp_across_bsz = comp_across_bsz.unsqueeze(0)
                                    
                        #             # combined_probe = (probe_out * 0.5 + self.last_batch_probe_out * (1 - 0.5)).to(probe_out.dtype)
                        #             print('combined_probe', combined_probe.shape, flush=True)
                                    
                        #             print('probe_out shape', probe_out.shape, flush=True)
                        #     else:
                        #         if self.last_batch_probe_out is None:
                        #             self.last_batch_probe_out = probe_out
                        #             probe_out = torch.zeros_like(probe_out)
                        #         else:
                        #         # self.last_batch_probe_out = self.last_batch_probe_out.to(probe_out.device)
                        #             # if 'asyncabs' in cfg['prune_method']:
                        #             #     abs_probe = torch.abs(probe_out).to(torch.float32)
                        #             #     abs_last_batch_probe = torch.abs(self.last_batch_probe_out).to(torch.float32)
                        #             #     sum_across_two_terms = abs_probe + abs_last_batch_probe
                        #             #     # proportion = abs_x / torch.sum(abs_x, dim=0, keepdim=True)
                        #             #     proportion = (abs_probe / (sum_across_two_terms + 1e-10)).to(abs_probe.dtype)
                        #             #     combined_probe = (probe_out * proportion + self.last_batch_probe_out * (1 - proportion)).to(probe_out.dtype)
                        #             #     probe_out, self.last_batch_probe_out = self.last_batch_probe_out, combined_probe
                        #             # else:
                        #             proportion = cfg['asyncratio']
                        #             combined_probe = (self.last_batch_probe_out * proportion + probe_out * (1 - proportion)).to(probe_out.dtype)
                        #             probe_out, self.last_batch_probe_out = self.last_batch_probe_out, combined_probe
                        #                 # proportion = 10
                        #             # print('proportion ', proportion, flush=True)
                                    # comp_across_bsz = torch.sum(x * proportion, dim=0)
                                    # comp_across_bsz = comp_across_bsz.unsqueeze(0)
                                    # combined_probe = (probe_out * proportion + self.last_batch_probe_out * (1 - proportion)).to(probe_out.dtype)
                                    # probe_out, self.last_batch_probe_out = self.last_batch_probe_out, combined_probe

                        
                        custom_duration = time.time() - time_start
                        # print('probe_duration', custom_duration, flush=True)
                        time_start = time.time()
                        if cfg['mode'] == 'sync':
                            if 'gate_proj' in cfg['cust_tgt_modules']:
                                gate_out = self.act_fn(self.gate_proj(x, out_dim_indices=probe_out_dim_indices))
                            # else:
                            #     # gate_out = self.act_fn(self.gate_proj(x))
                            #     gate_out = gate_out[..., probe_out_dim_indices]

                            if 'up_proj' in cfg['cust_tgt_modules']:
                                up_out = self.up_proj(x, out_dim_indices=probe_out_dim_indices)
                            
                            down_proj = self.down_proj(gate_out * up_out, in_dim_indices=probe_out_dim_indices)
                        else:                   
                            # print('here')      
                            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                        
                        if cfg['mode'] == 'asyncinter':
                            with torch.cuda.stream(cfg['cuda_stream1']):
                                _, _ = self.probe_process(x)
                        # else:
                        #     # up_out = self.up_proj(x)
                        #     up_out = up_out[..., probe_out_dim_indices]

                        # intermediate_output = 
                        
                            # self.down_proj.update_global_metric_score_distribution(intermediate_output[..., probe_out_dim_indices], probe_out_dim_indices)
                            # fill the probe predict for prune_out_dim_indices
                        # if 'fillpbmetric' in cfg['prune_method']:
                        #         # self.down_proj.update_global_metric_score_distribution(probe_out[..., prune_out_dim_indices], prune_out_dim_indices, batch_size=bsz, is_probe=True)
                        #         # Selecting specific dimensions
                        #     # .expand(bsz, -1, -1)
                        #     if 'runningmean' in cfg['prune_method']:
                        #         self.down_proj.update_global_metric_score_distribution(probe_out[..., prune_out_dim_indices], prune_out_dim_indices)
                        #     elif 'ema' in cfg['prune_method']:
                        #         if 'fillpbmetricoriginal' in cfg['prune_method']:
                        #             self.down_proj.update_global_metric_score_distribution_ema(probe_out[..., prune_out_dim_indices], prune_out_dim_indices, is_probe=True)
                        #         elif 'fillpbmetriccombine' in cfg['prune_method']:
                        #             self.down_proj.update_global_metric_score_distribution_ema(comined_probe_out[..., prune_out_dim_indices], prune_out_dim_indices, is_probe=True)
                        #         elif 'fillpbmetricub' in cfg['prune_method']:
                        #             full_inference = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                        #             # full_inference = torch.clamp(torch.linalg.vector_norm(full_inference, ord=2, dim=0) ** 2, min=cfg['data_type_min'], max=cfg['data_type_max']) / bsz_tensor
                        #             # full_selected = full_inference[..., probe_out_dim_indices]
                        #             full_pruned = full_inference[..., prune_out_dim_indices]
                        #             self.down_proj.update_global_metric_score_distribution_ema(full_pruned, prune_out_dim_indices)

                        # if 'halfsquareasync' in cfg['prune_method'] and 'savemetricseq' in cfg['prune_method']:
                        #     temp_norm_square = torch.clamp(torch.linalg.vector_norm(gate_out * up_out, ord=2, dim=0) ** 2, min=cfg['data_type_min'], max=cfg['data_type_max']) / bsz
                        #     self.last_batch_probe_out[..., probe_out_dim_indices] = temp_norm_square

                        
                        custom_duration = time.time() - time_start
                        # print('fll_batch_duration', custom_duration, flush=True)
                        return down_proj
                    elif 'calib' in cfg['prune_method'] and ('runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']):
                        # if cfg['logger_detailed_info'] == True:
                        #     print('input', x, flush=True)
                        bsz, _, _ = x.shape
                        time_start = time.time()

                        if cfg['mode'] == 'sync':
                            if torch.all(self.down_proj.get_global_metric_score_distribution() == 0):
                                out_dim_indices = torch.arange(self.intermediate_size, dtype=torch.long).to(device=x.device)
                            else:
                                out_dim_metric = cal_calib_prune_metric(self.down_proj.get_global_metric_score_distribution(), self.down_proj.weight.data, cfg['prune_metric'])

                                if 'globalratio' in cfg['prune_method']:
                                    out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.down_proj.pruning_ratio)
                                else:
                                    out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'])


                            temp = self.act_fn(self.gate_proj(x, out_dim_indices=out_dim_indices)) * self.up_proj(x, out_dim_indices=out_dim_indices)
                            down_proj = self.down_proj(temp, in_dim_indices=out_dim_indices)
                            if cfg['logger_detailed_info'] == True:
                                print('temp out', temp, flush=True)
                                print('down_proj out', down_proj, flush=True)
                            # down_proj = self.down_proj(self.act_fn(self.gate_proj(x, out_dim_indices=out_dim_indices)) * self.up_proj(x, out_dim_indices=out_dim_indices), in_dim_indices=out_dim_indices)
                        elif cfg['mode'] == 'asyncinter':
                            temp = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                            # cur_event = cfg[f'cuda_events_mlp_{self.layer_order}']
                            # cur_event.record(stream=cfg['cuda_default_stream'])
                            # torch.cuda.synchronize(cfg['cuda_default_stream'])
                            down_proj = self.down_proj(temp)
                            # down_proj = self.down_proj(temp, cur_event=cur_event)
                            if cfg['logger_detailed_info'] == True:
                                print('temp out', temp, flush=True)
                                print('down_proj out', down_proj, flush=True)

                            # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

                            with torch.cuda.stream(cfg['cuda_stream1']):
                                if torch.all(self.down_proj.get_global_metric_score_distribution() == 0):
                                    out_dim_indices = torch.arange(self.intermediate_size, dtype=torch.long).to(device=x.device)
                                else:
                                    out_dim_metric = cal_calib_prune_metric(self.down_proj.get_global_metric_score_distribution(), self.down_proj.weight.data, cfg['prune_metric'])
                                    # out_dim_metric = torch.arange(self.intermediate_size, dtype=torch.long).to(device=x.device)

                                    if 'globalratio' in cfg['prune_method']:
                                        out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.down_proj.pruning_ratio)
                                    else:
                                        out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'])

                                if cfg['logger_detailed_info'] == True:
                                    print('out_dim_indices', out_dim_indices, flush=True)
                                    print('prune_out_dim_indices', prune_out_dim_indices, flush=True)
                                self.gate_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                                self.up_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                                self.down_proj.prepare_async_weight(in_dim_indices=out_dim_indices)
                            # torch.cuda.synchronize()
                            # torch.cuda.synchronize(cfg['cuda_stream1'])
                        # if cfg['logger_detailed_info'] == True:
                        #     print('down_proj_out', down_proj, flush=True)
                        return down_proj
                    
                    # only calib (baselines)
                    elif 'calib' in cfg['prune_method'] or 'flap' in cfg['prune_method'] or 'wandasp' in cfg['prune_method']:
                        # no ema or runningmean
                        bsz, _, _ = x.shape
                        time_start = time.time()

                        if cfg['mode'] == 'asyncinter':
                            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

                            if cfg['cur_batch_index'] == 0:
                                # with torch.cuda.stream(cfg['cuda_stream1']):
                                if torch.all(self.down_proj.get_global_metric_score_distribution() == 0):
                                    out_dim_indices = torch.arange(self.intermediate_size, dtype=torch.long).to(device=x.device)
                                else:
                                    out_dim_metric = cal_calib_prune_metric(self.down_proj.get_global_metric_score_distribution(), self.down_proj.weight.data, cfg['prune_metric'])

                                    if 'globalratio' in cfg['prune_method']:
                                        out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.down_proj.pruning_ratio)
                                    else:
                                        out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'])

                                self.gate_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                                self.up_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                                self.down_proj.prepare_async_weight(in_dim_indices=out_dim_indices)
                        else:
                            raise ValueError('Invalid mode')
                        
                        # if self.cur_batch % 5 == 0:
                        #     if self.cur_batch == 0:
                        #         self.prev = probe_out_dim_indices
                        #     else:
                        #         set_first = set(self.prev.cpu().numpy())
                        #         set_second = set(probe_out_dim_indices.cpu().numpy())
                        #         intersection = set_first.intersection(set_second)
                        #         intersection_ratio = len(intersection) / len(set_first)
                        #         print(self.layer_order, 'intersection_ratio', intersection_ratio, flush=True)
                        #         self.prev = probe_out_dim_indices
                        # self.cur_batch += 1
                        custom_duration = time.time() - time_start
                        # print('fll_batch_duration', custom_duration, flush=True)
                        return down_proj
                    # elif 'runningmean' in cfg['prune_method'] and ('down_proj' in cfg['cust_tgt_modules'] or 'up_proj' in cfg['cust_tgt_modules'] or 'gate_proj' in cfg['cust_tgt_modules']):
                    #     bsz, _, _ = x.shape
                    #     time_start = time.time()
                    #     if torch.all(self.down_proj.get_global_metric_score_distribution() == 0):
                    #         probe_out_dim_indices = torch.arange(self.intermediate_size, dtype=torch.long).to(device=x.device)
                    #         # self.running_mean = torch.zeros(self.intermediate_size, dtype=x.dtype, device=x.device)
                    #         # self.running_mean_counter = torch.zeros(self.intermediate_size, dtype=torch.int32, device=x.device)
                    #     else:
                    #         probe_out_dim_metric = cal_calib_prune_metric(self.down_proj.get_global_metric_score_distribution(), self.down_proj.weight.data, cfg['prune_metric'])
                    #         probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, multiple)

                    #     temp = self.act_fn(self.gate_proj(x, probe_out_dim_indices=probe_out_dim_indices)) * self.up_proj(x, probe_out_dim_indices=probe_out_dim_indices)
                    #     # if 'runningmean' in cfg['prune_method']:
                    #         # print('runningmean', flush=True)
                    #         # print('probe_out_dim_indices', probe_out_dim_indices, flush=True)
                    #         # print('temp', temp, flush=True)
                    #         # self.down_proj.update_global_metric_score_distribution(temp, probe_out_dim_indices)
                    #         # Update the running_mean and running_mean_counter here
                    #         # self.running_mean[probe_out_dim_indices] *= self.running_mean_counter[probe_out_dim_indices] / (self.running_mean_counter[probe_out_dim_indices] + bsz)
                    #         # # Ensure the denominator is broadcastable; might need to unsqueeze to add a dimension for correct broadcasting
                    #         # norm_squared = torch.clamp(torch.linalg.vector_norm(temp, ord=2, dim=1) ** 2, min=cfg['data_type_min'], max=cfg['data_type_max'])
                    #         # denominator = (self.running_mean_counter[probe_out_dim_indices].unsqueeze(0) + bsz)
                    #         # # Update running mean
                    #         # self.running_mean[probe_out_dim_indices] += torch.sum(norm_squared / denominator, dim=0)
                    #         # self.running_mean_counter[probe_out_dim_indices] += bsz
                    #     kwargs['probe_in_dim_indices'] = probe_out_dim_indices
                    #     down_proj = self.down_proj(temp, **kwargs)
                    #     custom_duration = time.time() - time_start
                    #     print('fll_batch_duration', custom_duration, flush=True)
                    #     return down_proj
                    # else:
                    #     print('here')
                    #     mlp_duration_start = time.time()
                    #     time_start = time.time()
                    #     temp_gate = self.act_fn(self.gate_proj(x))
                    #     custom_duration = time.time() - time_start
                    #     print('custom_duration gate', custom_duration, flush=True)

                    #     time_start = time.time()
                    #     temp_up = self.up_proj(x)
                    #     custom_duration = time.time() - time_start
                    #     print('custom_duration up', custom_duration, flush=True)
                    #     time_start = time.time()
                    #     down_proj = self.down_proj(temp_gate * temp_up)
                    #     custom_duration = time.time() - time_start
                    #     print('custom_duration down', custom_duration, flush=True)
                    #     mlp_duration = time.time() - mlp_duration_start
                    #     print('mlp_duration', mlp_duration, flush=True)
                    #     del temp_gate, temp_up
                    #     return down_proj
            else:
                torch.cuda.synchronize()
                start_time = time.time()
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                torch.cuda.synchronize()
                end_time = time.time() - start_time
                print('mlp_full_batch_duration', end_time, flush=True)
                
                return down_proj

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_order: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.custom_duration = 0
        # self.cal_total_flops = True
        # self.default_num_heads = config.num_attention_heads
        # self.default_head_dim = self.hidden_size // self.default_num_heads
        # self.default_num_key_value_heads = config.num_key_value_heads

        self.pruning_module = HiddenRepresentationPruning(cfg, f'llama_attention_{layer_order}')

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # self.q_proj.cal_total_flops = True
        # self.k_proj.cal_total_flops = True
        # self.v_proj.cal_total_flops = True
        # self.o_proj.cal_total_flops = True

        self.layer_order = layer_order


        self.q_num_heads = None 
        self.k_num_heads = None
        self.v_num_heads = None
        self.q_head_dim = None
        self.k_head_dim = None
        self.v_head_dim = None
        # self.probe_qk_out_dim_indices = None
        # self.probe_qk_out_dim_indices_for_rope = None 
        # self.probe_vo_out_dim_indices = None

        self.attn_weights_indices = None
        # print("cfg['cust_tgt_modules']", cfg['cust_tgt_modules'])
        # if 'probe' in cfg['prune_name'] and 'each' in cfg['prune_name'] and ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']):
        #     self.head_dim = int((1 - cfg['prune_hyper']) * self.head_dim)
        #     print('self.head_dim', self.head_dim, flush=True)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.num_heads,
                self.head_dim,
                max_position_embeddings=cfg['seq_len'],
                base=self.rope_theta,
            )

            # self.inference_rotary_emb = copy.deepcopy(self.rotary_emb)
            # if 'WO' in cfg['prune_metric'] and ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']):
            #     if 'each' in cfg['prune_name']:
            #         # if number is odd, the RoPE will have problem
            #         head_dim = nearest_even_number((1 - cfg['prune_hyper']) * self.head_dim)
            #         # print('head_dimfor rotary', head_dim, flush=True)
            #         self.inference_rotary_emb = LlamaRotaryEmbedding(
            #             self.num_heads,
            #             head_dim,
            #             max_position_embeddings=self.max_position_embeddings,
            #             base=self.rope_theta,
            #         )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def probe_process(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs):
        # 1. generate probe
        # 2. run matrix multiplication
        # 3. calculate score
        # 4. extract metric

        bsz, q_len, _ = hidden_states.size()
        # generate probe
        # rank / mean / absnml
        if cfg['q_probe_ratio'] == cfg['k_probe_ratio'] and cfg['q_probe_ratio'] == cfg['v_probe_ratio']:
            probe, selected_indices = generate_probe(hidden_states, cfg[f'q_probe_ratio'])
        else:
            raise ValueError('q_probe_num should be equal to k_probe_num and v_probe_num for now')

        print('attn probe', probe.shape, flush=True)
        self.q_num_heads, self.k_num_heads, self.v_num_heads = self.num_heads, self.num_key_value_heads, self.num_key_value_heads
        self.q_head_dim, self.k_head_dim, self.v_head_dim = self.head_dim, self.head_dim, self.head_dim

        # copy orignal code and modify a little bit for probe pruning
        # currently does not implement for group attention, but it should work too
        query_states = self.q_proj(probe, cal_attn_probe_out_dim_metric=True)   
        key_states = self.k_proj(probe, cal_attn_probe_out_dim_metric=True)
        value_states = self.v_proj(probe, cal_attn_probe_out_dim_metric=True)
        
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        qk_prune_way = cfg['qk_prune_way']
        vo_prune_way = cfg['vo_prune_way']

        query_states = query_states.view(query_states.shape[0], q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(key_states.shape[0], q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(value_states.shape[0], q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        print('query_states', query_states.shape, flush=True)
        print('key_states', key_states.shape, flush=True)
        print('attn_weights', attn_weights.shape, flush=True)
        # print('attn_weights', attn_weights.shape, query_states.shape,flush=True)
        if attn_weights.size() != (cfg['k_probe_num'], self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(cfg['k_probe_num'], self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        print('attention_mask', attention_mask.shape, flush=True)
        probe_attn_mask = attention_mask[:cfg['k_probe_num'], ...]
        print('probe_attn_mask', probe_attn_mask.shape, flush=True)
        if probe_attn_mask is not None:
            if probe_attn_mask.size() != (cfg['k_probe_num'], 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(cfg['k_probe_num'], 1, q_len, kv_seq_len)}, but is {probe_attn_mask.size()}"
                )
            attn_weights = attn_weights + probe_attn_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # TODO: fix later
        # if 'delseq' in cfg['prune_info']:
        #     if 'calib' in cfg['prune_method'] or 'runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']:
        #         probe_out_dim_metric, comined_probe_out = cal_attn_weight_prune_metric(attn_output, value_states, cfg['prune_metric'], global_metric_score_distribution=self.o_proj.get_global_metric_score_distribution())
        #     else:
        #         probe_out_dim_metric, comined_probe_out = cal_attn_weight_prune_metric(attn_output, value_states, cfg['prune_metric'])

        #     if 'globalratio' in cfg['prune_method']:
        #         self.attn_weights_indices, _ , _ , _ = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, self.k_num_heads, cfg['seq_len'], 'each', 'attnweights', cfg['tc_multiple'], pruning_ratio=self.o_proj.pruning_ratio)
        #     else:
        #         # probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, multiple)
        #         self.attn_weights_indices, _ , _ , _ = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, self.k_num_heads, cfg['seq_len'], 'each', 'attnweights', cfg['tc_multiple'])


        # if cfg['k_probe_num'] > cfg['v_probe_num']:
        #     value_states = value_states.repeat_interleave(cfg['k_probe_num']// cfg['v_probe_num'], dim=0)
        # else:
        #     attn_weights = attn_weights.repeat_interleave(cfg['v_probe_num']// cfg['k_probe_num'], dim=0)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (max(cfg['k_probe_num'], cfg['v_probe_num']), self.v_num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(max(cfg['k_probe_num'], cfg['v_probe_num']), self.v_num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(max(cfg['k_probe_num'], cfg['v_probe_num']), q_len, self.hidden_size)
        
        if 'calib' in cfg['prune_method'] or 'runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']:
            probe_out_dim_metric, comined_probe_out = cal_prune_metric(attn_output, self.o_proj.weight.data, cfg['prune_metric'], global_metric_score_distribution=self.o_proj.get_global_metric_score_distribution(), selected_indices=selected_indices)
        else:
            probe_out_dim_metric, comined_probe_out = cal_prune_metric(attn_output, self.o_proj.weight.data, cfg['prune_metric'])

        if 'globalratio' in cfg['prune_method']:
            probe_vo_out_dim_indices, probe_vo_out_dim_indices_for_rope, self.v_num_heads, self.v_head_dim = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, self.v_num_heads, self.v_head_dim, vo_prune_way, 'vo', cfg['tc_multiple'], pruning_ratio=self.o_proj.pruning_ratio)
        else:
            # probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, multiple)
            probe_vo_out_dim_indices, probe_vo_out_dim_indices_for_rope, self.v_num_heads, self.v_head_dim = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, self.v_num_heads, self.v_head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])
            if probe_vo_out_dim_indices is not None:
                print('probe_vo_out_dim_indices', probe_vo_out_dim_indices.shape, flush=True)
        
        if vo_prune_way is None:
            probe_vo_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
        else:
            pass

        if qk_prune_way is None:
            probe_qk_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
            probe_qk_out_dim_indices_for_rope = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
            # print('probe_qk_out_dim_indices', probe_qk_out_dim_indices, flush=True)
        else:
            
            # TODO: fix later, selfqk
            if 'selfqk' in qk_prune_way:
                pass
            else:
                self.q_num_heads, self.k_num_heads = self.v_num_heads, self.v_num_heads
                self.q_head_dim, self.k_head_dim = self.v_head_dim, self.v_head_dim
                probe_qk_out_dim_indices = probe_vo_out_dim_indices
                probe_qk_out_dim_indices_for_rope = probe_vo_out_dim_indices_for_rope
                # print('probe_qk_out_dim_indices', self.probe_qk_out_dim_indices.shape, flush=True)
                print('self.q_num_heads', self.q_num_heads, flush=True)

        self.q_proj.prepare_async_weight(out_dim_indices=probe_qk_out_dim_indices)
        self.k_proj.prepare_async_weight(out_dim_indices=probe_qk_out_dim_indices)
        self.o_proj.prepare_async_weight(in_dim_indices=probe_vo_out_dim_indices)
        self.o_proj.prepare_async_weight(in_dim_indices=probe_vo_out_dim_indices)
        # attn_output, attn_weights, past_key_value
        return probe_qk_out_dim_indices, probe_qk_out_dim_indices_for_rope, probe_vo_out_dim_indices, attn_weights, attn_output, past_key_value


    def attention_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs
        ):
        time_start = time.time()
        # full inference
        bsz, q_len, _ = hidden_states.size()
        # print('attn hiddenstate device', hidden_states.device)
        # print('q proj device', self.q_proj.weight.device)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # print('query_states key_states after rotary', query_states, key_states, flush=True)
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # key_states: bsz, self.num_key_value_heads, q_len, self.head_dim -> bsz, self.num_key_value_heads, self.head_dim, q_len
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
        # upcast attention to fp32
        # attn_weights: bsz, self.num_heads, q_len, q_len
        # attn_weights: 1, self.num_heads, q_len, q_len
        # print(torch.cuda.memory_summary())
        # print('attn_weights', attn_weights.shape, flush=True)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # value_states: bsz, self.num_key_value_heads, q_len, self.head_dim
        # value_states: 1, self.num_key_value_heads, q_len, self.head_dim
        attn_output = torch.matmul(attn_weights, value_states)
        # print('attn_output after value', attn_output)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        custom_duration = time.time() - time_start
        # print('custom_duration llama attention', custom_duration, flush=True)
        return attn_output, attn_weights, past_key_value


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        # if 'llama-2-70b' in cfg['model_name'] and self.modify_kv_for_llama_2_70b == True:             
        #     if self.k_proj.weight.data.size(0) == self.num_key_value_heads * self.head_dim:
        #         temp_weight_data = self.k_proj.weight.data.repeat_interleave(self.num_key_value_groups, dim=0, output_size=self.num_heads * self.head_dim)
        #         temp_weight_data = temp_weight_data.type(self.k_proj.weight.data.dtype)
        #         self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        #         self.k_proj.weight = nn.Parameter(temp_weight_data)
        #         self.k_proj.cal_total_flops = True
        #         del temp_weight_data
        #         torch.cuda.empty_cache()
        #     if self.v_proj.weight.data.size(0) == self.num_key_value_heads * self.head_dim:
        #         temp_weight_data = self.v_proj.weight.data.repeat_interleave(self.num_key_value_groups, dim=0, output_size=self.num_heads * self.head_dim)
        #         temp_weight_data = temp_weight_data.type(self.v_proj.weight.data.dtype)
        #         self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        #         self.v_proj.weight = nn.Parameter(temp_weight_data)
        #         self.v_proj.cal_total_flops = True
        #         del temp_weight_data
        #         torch.cuda.empty_cache()

        #     self.num_key_value_heads = self.num_heads
        #     self.num_key_value_groups = 1
        #     self.modify_kv_for_llama_2_70b = False
        if ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']) and self.layer_order > cfg['skip_layers']:
            if cfg['calibration_stage'] == True:
                return self.attention_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
            elif cfg['calibration_stage'] == False :
                bsz, q_len, _ = hidden_states.size()
                qk_prune_way = cfg['qk_prune_way']
                vo_prune_way = cfg['vo_prune_way']
                if 'probe' in cfg['prune_method']:
                    if cfg['mode'] == 'sync':
                        probe_qk_out_dim_indices, probe_qk_out_dim_indices_for_rope, probe_vo_out_dim_indices, attn_weights, attn_output, past_key_value = self.probe_process(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
                        if cfg['onlyprobe'] == True:
                            # attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
                            # # match the shape, and will not count the flops for this part
                            # attn_output = self.o_proj(attn_output, cal_attn_probe_out_dim_metric=True)
                            attn_output = torch.zeros((cfg['batch_size'], cfg['seq_len'], self.hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
                            if not output_attentions:
                                attn_weights = None
                            return attn_output, attn_weights, past_key_value
                    elif cfg['mode'] == 'asyncintra':
                        # if not, do full inference
                        if 'input_layernorm_mlp_residual' in kwargs:
                            # print('post_layernorm_attn_residual', flush=True)
                            _, _, _, _, _, _ = self.probe_process(kwargs['input_layernorm_mlp_residual'], attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
                            return
                    
                    # --------------------------------------
                    #full inference with adding some info to layer input
                    bsz, q_len, _ = hidden_states.size()


                    query_states = self.q_proj(hidden_states, out_dim_indices=probe_qk_out_dim_indices)
                    key_states = self.k_proj(hidden_states, out_dim_indices=probe_qk_out_dim_indices)
                    value_states = self.v_proj(hidden_states, out_dim_indices=probe_vo_out_dim_indices)

                    # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                    # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                    # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                    query_states = query_states.view(bsz, q_len, self.q_num_heads, self.q_head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, self.k_num_heads, self.k_head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, self.v_num_heads, self.v_head_dim).transpose(1, 2)

                    kv_seq_len = key_states.shape[-2]
                    if past_key_value is not None:
                        kv_seq_len += past_key_value[0].shape[-2]
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

                    if qk_prune_way is not None and 'each' in qk_prune_way:
                        query_states, key_states = apply_rotary_pos_emb_for_prune_each_head(query_states, key_states, cos, sin, position_ids, self.probe_qk_out_dim_indices_for_rope)
                    else:
                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

                    if past_key_value is not None:
                        key_states = torch.cat([past_key_value[0], key_states], dim=2)
                        value_states = torch.cat([past_key_value[1], value_states], dim=2)

                    past_key_value = (key_states, value_states) if use_cache else None
                    key_states = repeat_kv(key_states, self.num_key_value_groups)
                    value_states = repeat_kv(value_states, self.num_key_value_groups)

                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                    if attn_weights.size() != (bsz, self.k_num_heads, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                            f" {attn_weights.size()}"
                        )

                    if attention_mask is not None:
                        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                            raise ValueError(
                                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                            )
                        attn_weights = attn_weights + attention_mask
                        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

                    prev_attention_weight = attn_weights.clone()
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                    # attn_output = torch.matmul(attn_weights, value_states)
                    # if attn_output.size() != (bsz, self.v_num_heads, q_len, self.v_head_dim):
                    #     raise ValueError(
                    #         f"`attn_output` should be of size {(bsz, self.v_num_heads, q_len, self.v_head_dim)}, but is"
                    #         f" {attn_output.size()}"
                    #     )
                    # attn_output = attn_output.transpose(1, 2).contiguous()
                    # attn_output = attn_output.reshape(bsz, q_len, self.v_num_heads * self.v_head_dim)

                    # probe_out_dim_metric, comined_probe_out = cal_prune_metric(attn_output, self.o_proj.weight.data, cfg['prune_metric'])
                    # self.probe_vo_out_dim_indices, self.probe_vo_out_dim_indices_for_rope, self.v_num_heads, self.v_head_dim = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, self.v_num_heads, self.v_head_dim, 'each', 'vo', cfg['tc_multiple'])
                    # print(self.v_num_heads, self.v_head_dim, flush=True)
                    # if self.probe_vo_out_dim_indices is not None:
                    #     print(self.probe_vo_out_dim_indices.shape, flush=True)
                    # value_states = self.v_proj(hidden_states, out_dim_indices=self.probe_vo_out_dim_indices)
                    # print(value_states.shape, flush=True)
                    # value_states = value_states.view(bsz, q_len, self.v_num_heads, self.v_head_dim).transpose(1, 2)

                    # if 'delseq' in cfg['probe_info'] and cfg['cur_batch_index'] != 0:
                    if 'delseq' in cfg['probe_info']:
                        # temp_attn_weights = attn_weights
                        # attention_mask = attention_mask
                        # print('attention_mask', attention_mask)
                        converted_mask = attention_mask.clone()
                        converted_mask[converted_mask == 0] = 1
                        converted_mask[converted_mask < 0] = 0

                        # Sum over the 0th (batch size) and 2nd (query length) dimensions
                        summed_mask = converted_mask.sum(dim=(0, 2)) / bsz
                        # print('summed_mask', summed_mask, flush=True)
                        # / torch.sqrt(summed_mask)
                        # attn_weights_metric = (torch.clamp(torch.linalg.vector_norm(attn_weights, ord=2, dim=(0, 2)).reshape((1, self.num_heads, -1, 1)), max=cfg['data_type_max']) * torch.abs(value_states)).sum(axis=(0, -1))

                        # # each head, different seq
                        # norm_square = torch.clamp(torch.linalg.vector_norm(attn_weights, ord=2, dim=(0, 2)).reshape((1, self.num_heads, -1, 1)) ** 2, max=cfg['data_type_max'])
                        # attn_weights_metric = torch.sqrt((norm_square * torch.pow(value_states, 2)).sum(axis=(0, -1))).clamp(max=cfg['data_type_max'])
                        # # attn_weights_metric = (norm_square * torch.pow(value_states, 2)).sum(axis=(0, -1)).clamp(max=cfg['data_type_max'])
                        # # self.probe_vo_out_dim_indices, self.probe_vo_out_dim_indices_for_rope, self.v_num_heads, self.v_head_dim = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, self.v_num_heads, self.v_head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])
                        # _, attn_weights_indices, _, _ = self.pruning_module.sort_probe_attn_metric(attn_weights_metric, attn_weights.shape[1], attn_weights.shape[-1], 'each', 'delseq', cfg['tc_multiple'])
                        
                        # # _, attn_weights_indices, _, _ = self.pruning_module.sort_probe_attn_metric(attn_weights_metric, attn_weights.shape[1], self.head_dim, 'each', 'delseq', cfg['tc_multiple'])

                        # attn_weights_indices_expand = attn_weights_indices.unsqueeze(0).unsqueeze(2).expand(bsz, self.k_num_heads, q_len, -1)
                        # # attn_weights = torch.gather(attn_weights, -1, attn_weights_indices_expand)
                        # print('del seq prev_attention_weight', prev_attention_weight.shape, flush=True)
                        # prev_attention_weight = torch.gather(prev_attention_weight, -1, attn_weights_indices_expand)
                        # attn_weights = nn.functional.softmax(prev_attention_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        # value_states_indices_expand = attn_weights_indices.unsqueeze(0).unsqueeze(3).expand(bsz, self.v_num_heads, -1, self.v_head_dim)
                        # value_states = torch.gather(value_states, -2, value_states_indices_expand)
                        # # print('del seq attn_weights', attn_weights.shape, flush=True)
                        # # print('sel seq value_states', value_states.shape, flush=True)
                        # sorted_indices, sorted_positions = torch.sort(attn_weights_indices)


                        # each head, same seq
                        norm_square = torch.clamp(torch.linalg.vector_norm(attn_weights, ord=2, dim=(0, 2)).reshape((1, self.num_heads, -1, 1)) ** 2, max=cfg['data_type_max'])

                        # / (converted_mask.sum(dim=(0, 2)) / bsz)
                        norm_square = torch.clamp(torch.linalg.vector_norm(attn_weights, ord=2, dim=(0, 2)).reshape((1, self.num_heads, -1, 1)) ** 2 , max=cfg['data_type_max'])
                        attn_weights_metric = torch.sqrt((norm_square * torch.pow(value_states, 2)).sum(axis=(0, -1))).clamp(max=cfg['data_type_max'])


                        # norm_square = torch.clamp(torch.linalg.vector_norm(attn_weights, ord=2, dim=(0, 1, 2)).reshape((1, 1, -1, 1)) ** 2 , max=cfg['data_type_max'])
                        # attn_weights_metric = (norm_square * torch.pow(value_states, 2)).sum(axis=(0, -1)).clamp(max=cfg['data_type_max'])
                        attn_weights_metric = torch.linalg.norm(attn_weights_metric, ord=2, dim=0)
                        # attn_weights_metric = torch.sum(attn_weights_metric, dim=0).clamp(max=cfg['data_type_max'])
                        probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(attn_weights_metric, cfg['tc_multiple'])
                        # self.probe_vo_out_dim_indices, self.probe_vo_out_dim_indices_for_rope, self.v_num_heads, self.v_head_dim = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, self.v_num_heads, self.v_head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])
                        # attn_weights = attn_weights[:, :, :, probe_out_dim_indices]
                        attn_weights = prev_attention_weight[:, :, :, probe_out_dim_indices]
                        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        value_states = value_states[:, :, probe_out_dim_indices, :]
                        


                    # if 'delseq' in cfg['probe_info'] and cfg['cur_batch_index'] != 0:
                    #     attn_weights_indices_expand = self.attn_weights_indices.unsqueeze(0).unsqueeze(2).expand(bsz, self.k_num_heads, q_len, -1)
                    #     value_states_indices_expand = self.attn_weights_indices.unsqueeze(0).unsqueeze(3).expand(bsz, self.v_num_heads, -1, self.v_head_dim)

                    attn_output = torch.matmul(attn_weights, value_states)
                    if attn_output.size() != (bsz, self.v_num_heads, q_len, self.v_head_dim):
                        raise ValueError(
                            f"`attn_output` should be of size {(bsz, self.v_num_heads, q_len, self.v_head_dim)}, but is"
                            f" {attn_output.size()}"
                        )
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, self.v_num_heads * self.v_head_dim)

                    attn_output = self.o_proj(attn_output, in_dim_indices=probe_vo_out_dim_indices)

                    if cfg['mode'] == 'asyncinter':
                        with torch.cuda.stream(cfg['cuda_stream1']):
                            _, _, _, _, _, _ = self.probe_process(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
                            
                    if not output_attentions:
                        attn_weights = None
                    
                    return attn_output, attn_weights, past_key_value
                elif 'calib' in cfg['prune_method'] and ('runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']):
                    time_start = time.time()
                    bsz, q_len, _ = hidden_states.size()

                    q_num_heads, k_num_heads, v_num_heads = self.num_heads, self.num_key_value_heads, self.num_key_value_heads
                    q_head_dim, k_head_dim, v_head_dim = self.head_dim, self.head_dim, self.head_dim

                    qk_prune_way = cfg['qk_prune_way']
                    vo_prune_way = cfg['vo_prune_way']
                    
                    probe_qk_out_dim_indices = None
                    probe_vo_out_dim_indices = None
                    if cfg['mode'] == 'sync':
                        if vo_prune_way is None:
                            probe_vo_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                        else:
                            if torch.all(self.o_proj.get_global_metric_score_distribution() == 0):
                                probe_vo_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                            else:
                                probe_out_dim_metric = cal_calib_prune_metric(self.o_proj.get_global_metric_score_distribution(), self.o_proj.weight.data, cfg['prune_metric'])
                                if 'globalratio' in cfg['prune_method']:
                                    probe_vo_out_dim_indices, probe_vo_out_dim_indices_for_rope, v_num_heads, v_head_dim = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, v_num_heads, v_head_dim, vo_prune_way, 'vo', cfg['tc_multiple'], pruning_ratio=self.o_proj.pruning_ratio)
                                else:
                                    # probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, multiple)
                                    probe_vo_out_dim_indices, probe_vo_out_dim_indices_for_rope, v_num_heads, v_head_dim = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, v_num_heads, v_head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])
                        
                        if qk_prune_way is None:
                            probe_qk_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                            probe_qk_out_dim_indices_for_rope = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                            # print('probe_qk_out_dim_indices', probe_qk_out_dim_indices, flush=True)
                        else:
                            
                            # TODO: fix later, selfqk
                            if 'selfqk' in qk_prune_way:
                                pass
                            else:
                                # default, follow vo
                                q_num_heads, k_num_heads = v_num_heads, v_num_heads
                                q_head_dim, k_head_dim = v_head_dim, v_head_dim
                                probe_qk_out_dim_indices = probe_vo_out_dim_indices
                                probe_qk_out_dim_indices_for_rope = probe_vo_out_dim_indices_for_rope
                        
                    query_states = self.q_proj(hidden_states, out_dim_indices=probe_qk_out_dim_indices)
                    key_states = self.k_proj(hidden_states, out_dim_indices=probe_qk_out_dim_indices)
                    value_states = self.v_proj(hidden_states, out_dim_indices=probe_vo_out_dim_indices)
                
                    query_states = query_states.view(bsz, q_len, q_num_heads, q_head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, k_num_heads, k_head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, v_num_heads, v_head_dim).transpose(1, 2)

                    kv_seq_len = key_states.shape[-2]
                    if past_key_value is not None:
                        kv_seq_len += past_key_value[0].shape[-2]
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                    
                    if qk_prune_way is not None and 'each' in qk_prune_way:
                        query_states, key_states = apply_rotary_pos_emb_for_prune_each_head(query_states, key_states, cos, sin, position_ids, probe_qk_out_dim_indices_for_rope)
                    else:
                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
                    # print('query_states key_states after rotary', query_states, key_states, flush=True)
                    if past_key_value is not None:
                        # reuse k, v, self_attention
                        key_states = torch.cat([past_key_value[0], key_states], dim=2)
                        value_states = torch.cat([past_key_value[1], value_states], dim=2)

                    past_key_value = (key_states, value_states) if use_cache else None
                    key_states = repeat_kv(key_states, self.num_key_value_groups)
                    value_states = repeat_kv(value_states, self.num_key_value_groups)

                    # key_states: bsz, self.num_key_value_heads, q_len, self.head_dim -> bsz, self.num_key_value_heads, self.head_dim, q_len
                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(k_head_dim)

                    if attn_weights.size() != (bsz, k_num_heads, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention weights should be of size {(bsz, k_num_heads, q_len, kv_seq_len)}, but is"
                            f" {attn_weights.size()}"
                        )

                    if attention_mask is not None:
                        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                            raise ValueError(
                                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                            )
                        attn_weights = attn_weights + attention_mask
                        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
                    # upcast attention to fp32
                    # attn_weights: bsz, self.num_heads, q_len, q_len
                    # attn_weights: 1, self.num_heads, q_len, q_len
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    # value_states: bsz, self.num_key_value_heads, q_len, self.head_dim
                    # value_states: 1, self.num_key_value_heads, q_len, self.head_dim
                    print('attn_weights', attn_weights.shape, flush=True)
                    print('value_states', value_states.shape, flush=True)
                    attn_output = torch.matmul(attn_weights, value_states)
                    # print('attn_output after value', attn_output)

                    if attn_output.size() != (bsz, v_num_heads, q_len, v_head_dim):
                        raise ValueError(
                            f"`attn_output` should be of size {(bsz, v_num_heads, q_len, v_head_dim)}, but is"
                            f" {attn_output.size()}"
                        )

                    attn_output = attn_output.transpose(1, 2).contiguous()

                    # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                    attn_output = attn_output.reshape(bsz, q_len, v_num_heads * v_head_dim)

                    attn_output = self.o_proj(attn_output, in_dim_indices=probe_vo_out_dim_indices)
                    if not output_attentions:
                        attn_weights = None

                    custom_duration = time.time() - time_start
                    # print('custom_duration llama attention', custom_duration, flush=True)
                        
                    
                    if cfg['mode'] == 'asyncinter':
                        with torch.cuda.stream(cfg['cuda_stream1']):
                            if vo_prune_way is None:
                                vo_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                            else:
                                if torch.all(self.o_proj.get_global_metric_score_distribution() == 0):
                                    vo_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                                else:
                                    # TODO: deal with rope
                                    out_dim_metric = cal_calib_prune_metric(self.o_proj.get_global_metric_score_distribution(), self.o_proj.weight.data, cfg['prune_metric'])
                                    if 'globalratio' in cfg['prune_method']:
                                        vo_out_dim_indices, probe_vo_out_dim_indices_for_rope, v_num_heads, v_head_dim = self.pruning_module.sort_probe_attn_metric(out_dim_metric, v_num_heads, v_head_dim, vo_prune_way, 'vo', cfg['tc_multiple'], pruning_ratio=self.o_proj.pruning_ratio)
                                    else:
                                        # probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, multiple)
                                        vo_out_dim_indices, probe_vo_out_dim_indices_for_rope, v_num_heads, v_head_dim = self.pruning_module.sort_probe_attn_metric(out_dim_metric, v_num_heads, v_head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])
                            
                            if qk_prune_way is None:
                                qk_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                                qk_out_dim_indices_for_rope = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                                # print('probe_qk_out_dim_indices', probe_qk_out_dim_indices, flush=True)
                            else:
                                
                                if 'selfqk' in qk_prune_way:
                                    pass
                                else:
                                    # default, follow vo
                                    q_num_heads, k_num_heads = v_num_heads, v_num_heads
                                    q_head_dim, k_head_dim = v_head_dim, v_head_dim
                                    qk_out_dim_indices = vo_out_dim_indices
                                    probe_qk_out_dim_indices_for_rope = probe_vo_out_dim_indices_for_rope
                            
                            self.q_proj.prepare_async_weight(out_dim_indices=qk_out_dim_indices)
                            self.k_proj.prepare_async_weight(out_dim_indices=qk_out_dim_indices)
                            self.o_proj.prepare_async_weight(in_dim_indices=vo_out_dim_indices)
                            self.o_proj.prepare_async_weight(in_dim_indices=vo_out_dim_indices)
                    return attn_output, attn_weights, past_key_value
                elif 'calib' in cfg['prune_method'] or 'flap' in cfg['prune_method'] or 'wandasp' in cfg['prune_method']:
                    pass
        else:
            return self.attention_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)



class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_order: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            LlamaAttention(config=config, layer_order=layer_order)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else LlamaFlashAttention2(config=config)
        )
        self.mlp = LlamaMLP(config, layer_order)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_order = layer_order

    def check_asyncintra_mlp(self):
        if ('down_proj' in cfg['cust_tgt_modules'] or 'up_proj' in cfg['cust_tgt_modules'] or 'gate_proj' in cfg['cust_tgt_modules']) \
            and self.layer_order > cfg['skip_layers'] \
            and cfg['calibration_stage'] == False \
            and cfg['mode'] == 'asyncintra' \
            and 'probe' in cfg['prune_method']:
            return True
        return False
    
    def check_asyncintra_attention(self):
        if ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']) \
            and self.layer_order > cfg['skip_layers'] \
            and cfg['calibration_stage'] == False \
            and cfg['mode'] == 'asyncintra' \
            and 'probe' in cfg['prune_method']:
            return True
        return False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        


        # torch.cuda.nvtx.range_push("zzzzzzz")
        start_time = time.time()
        # probe_gate = nml_process(hidden_states, cfg['gate_probe_num'], cfg['gate_probe_size'])
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('probe_gate_duration', end_time-start_time, flush=True)

        # tensor = torch.randn(100, 512, 4096, dtype=torch.float16, device='cuda')
        # probe_gate = nml_process(tensor, cfg['gate_probe_num'], cfg['gate_probe_size'])
        # torch.cuda.synchronize()
        # end_time_2 = time.time()
        # print('probe_gate_duration2', end_time_2-end_time, flush=True)

        # probe_gate = mean_process(tensor, cfg['gate_probe_num'], cfg['gate_probe_size'])
        # torch.cuda.synchronize()
        # end_time_3= time.time()
        # print('probe_gate_duration3', end_time_3-end_time_2, flush=True)

        # probe_gate = optimized_nml_process(tensor, cfg['gate_probe_num'], cfg['gate_probe_size'])
        # torch.cuda.synchronize()
        # end_time_4 = time.time()
        # print('probe_gate_duration4', end_time_4-end_time_3, flush=True)

        residual = hidden_states
        
        
        def probe_mlp_inf(result_dict):
            if self.check_asyncintra_mlp():
                with torch.cuda.stream(cfg['cuda_stream1']):
                    post_layernorm_attn_residual = self.post_attention_layernorm(residual)
                    if 'resinfo' in cfg['prune_method']: 
                        result_dict['post_layernorm_attn_residual'] = post_layernorm_attn_residual
                    self.mlp(hidden_states, post_layernorm_attn_residual=post_layernorm_attn_residual)
        
        
        
        def full_attn_inf(result_dict):
            try:
                hidden_states = self.input_layernorm(residual)
                hidden_states, self_attn_weights, present_key_value = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = residual + hidden_states
                result_dict['hidden_states'] = hidden_states
                print("Thread completed and added 'hidden_states'")
            except Exception as e:
                print(f"Error in full_attn_inf: {e}")
                
                traceback.print_exc() 

        # Wait for both threads to complete
        # thread1.join()
        # thread2.join()
        # Create threads
        probe_mlp_inf_result = {}
        full_attn_inf_result = {}
        thread1 = threading.Thread(target=full_attn_inf, args=(full_attn_inf_result,))
        thread2 = threading.Thread(target=probe_mlp_inf, args=(probe_mlp_inf_result,))

        # Start the threads
        thread1.start()
        thread2.start()
        
        

        if 'resinfo' in cfg['prune_method'] and self.check_asyncintra_mlp():
            self.attn_sign_match_percentage, self.attn_l1_diff_percentage, self.attn_cosine_similarity = cal_res_hidden_state_diff(hidden_states, probe_mlp_inf_result['post_layernorm_attn_residual'])
        
        # Fully Connected
        # hidden_states = full_attn_inf_result['hidden_states']
        # residual = hidden_states
        thread1.join()
        thread2.join()


        hidden_states = full_attn_inf_result['hidden_states']

        # hidden_states = self.input_layernorm(residual)
        # hidden_states, self_attn_weights, present_key_value = self.self_attn(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=past_key_value,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        # )
        # hidden_states = residual + hidden_states
        residual = hidden_states
        # print('hiddenstateshape', hidden_states.shape)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.check_asyncintra_attention():
            with torch.cuda.stream(cfg['cuda_stream1']):
                input_layernorm_mlp_residual = kwargs['next_layer'].input_layernorm(residual)
                _, _, _ = kwargs['next_layer'].self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    input_layernorm_mlp_residual=input_layernorm_mlp_residual
                )
        hidden_states = residual + hidden_states

        if 'resinfo' in cfg['prune_method'] and self.check_asyncintra_attention():
            self.mlp_sign_match_percentage, self.mlp_l1_diff_percentage, self.mlp_cosine_similarity = cal_res_hidden_state_diff(hidden_states, input_layernorm_mlp_residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        custom_duration = time.time() - start_time
        # torch.cuda.nvtx.range_pop()
        # print('custom_duration decoder layer', custom_duration, flush=True)
        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_layers_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        hidden_layers = config.num_hidden_layers
        # hidden_layers = 1
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_order) for layer_order in range(hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self.inference_time = 0

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            # print('attention_mask', attention_mask.shape, flush=True)
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False


        
            
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # print('idx decoder_layer', idx, decoder_layer, flush=True)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                torch.cuda.synchronize(cfg['cuda_default_stream']) 
                start_time = time.time()
                # start_event = torch.cuda.Event(enable_timing=True)
                # stop_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.nvtx.range_push("layer{}".format(idx))
                # start_event.record()
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    next_layer=self.layers[idx + 1] if idx + 1 < len(self.layers) else None,
                )

                # Record the end time
                # stop_event.record()

                # Wait for all the operations to complete
                 # Wait for the events to be recorded!

                # Calculate the elapsed time
                # elapsed_time_ms = start_event.elapsed_time(stop_event)
                torch.cuda.synchronize(cfg['cuda_default_stream']) 
                duration = time.time() - start_time
                torch.cuda.nvtx.range_pop()
                print(f"layer: {idx}, Elapsed time: {duration} milliseconds")
                print(f'layerdevice, {decoder_layer.mlp.gate_proj.weight.device}')
                if idx > cfg['skip_layers']:
                    self.inference_time += duration
               
            
            

            # print('layer_outputs', layer_outputs, flush=True)
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # if cfg['calibration_stage'] == False:
            #     if idx == len(self.layers)//2 - 1 or idx == len(self.layers) - 1:
            #         with torch.cuda.stream(cfg['cuda_stream1']):
            #             if idx == len(self.layers)//2 - 1:
            #                 finished_layers = list(range(cfg['skip_layers'] + 1, len(self.layers)//2))
            #             else:
            #                 finished_layers = list(range(len(self.layers)//2, len(self.layers)))
            #             for layer_order in finished_layers:
            #                 # print('layer_order', layer_order, flush=True)
            #                 cur_mlp = self.layers[layer_order].mlp
            #                 # attributes = dir(cur_mlp.down_proj)
            #                 # non_method_attributes = [attr for attr in attributes if not callable(getattr(cur_mlp.down_proj, attr)) and not attr.startswith('_')]
            #                 # print(non_method_attributes)
            #                 if cur_mlp.down_proj.async_interbatch_metric_index != cfg['cur_batch_index']:
            #                     print('sync metric step start', cur_mlp.down_proj.async_interbatch_metric_index)
            #                     print('sync metric step start batch', cfg['cur_batch_index'])
            #                     torch.cuda.synchronize(cfg['cuda_default_stream'])
            #                     print('wait sync metric step end', cur_mlp.down_proj.async_interbatch_metric_index)
            #                 if torch.all(cur_mlp.down_proj.get_global_metric_score_distribution() == 0):
            #                     out_dim_indices = torch.arange(cur_mlp.intermediate_size, dtype=torch.long).to(device=cur_mlp.down_proj.weight.data.device)
            #                 else:
            #                     out_dim_metric = cal_calib_prune_metric(cur_mlp.down_proj.get_global_metric_score_distribution(), cur_mlp.down_proj.weight.data, cfg['prune_metric'])

            #                     if 'globalratio' in cfg['prune_method']:
            #                         out_dim_indices, prune_out_dim_indices = cur_mlp.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'], pruning_ratio=cur_mlp.down_proj.pruning_ratio)
            #                     else:
            #                         out_dim_indices, prune_out_dim_indices = cur_mlp.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'])

            #                 if cfg['logger_detailed_info'] == True:
            #                     print('out_dim_indices', out_dim_indices, flush=True)
            #                     print('prune_out_dim_indices', prune_out_dim_indices, flush=True)
            #                 cur_mlp.gate_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
            #                 cur_mlp.up_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
            #                 cur_mlp.down_proj.prepare_async_weight(in_dim_indices=out_dim_indices)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_layers_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print('input_ids', input_ids.shape, flush=True)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # shift_logits = logits[..., :-1, :]
            # shift_labels = labels[..., 1:]
            # print('shift_logits', shift_logits, shift_logits.shape, flush=True)
            # print('shift_labels', shift_labels, shift_labels.shape, flush=True)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # loss = torch.tensor(5, device=logits.device)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )