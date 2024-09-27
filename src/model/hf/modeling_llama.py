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
import numpy as np
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
# from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
# 4.35.0 transformers verions has problem with the mask -> will get float('-inf'), use 4.37 version
from .attentionmask import AttentionMaskConverter, _prepare_4d_causal_attention_mask
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
from ..pruning_module import HiddenRepresentationPruning
from torch.nn.functional import cosine_similarity
from .utils import generate_probe, cal_res_hidden_state_diff, check_nan_inf, get_next_layer  
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

    def forward(self, hidden_states, keepdatatype=False):
        input_dtype = hidden_states.dtype
        if keepdatatype:
            variance = torch.clamp(hidden_states.pow(2).mean(-1, keepdim=True), max=cfg['data_type_max'])
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        else:
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
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

    cos = torch.gather(cos, -1, index_tensor)
    sin = torch.gather(sin, -1, index_tensor)
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
        self.act_fn = ACT2FN[config.hidden_act]

        self.layer_order = layer_order
        self.pruning_module = HiddenRepresentationPruning(cfg, f'llama_mlp_{layer_order}', config)

        self.probe_out_dim_indices = None
  
    def probe_process(self, x, **kwargs):
        # 1. generate probeW
        # 2. run matrix multiplication
        # 3. calculate score
        # 4. extract metric

        # generate probe
        # rank / mean / absnml
        cur_batch_seq_len = x.size(1)
        if cfg['gate_probe_ratio'] == cfg['up_probe_ratio']:
            if 'respick' in cfg['prune_method']:
                residual_for_probe = kwargs['respick']
            else:
                residual_for_probe = None
            
            probe, bsz_selected_indices, seq_selected_indices = generate_probe(x, cfg['gate_probe_ratio'], residual_for_probe)
        else:
            raise ValueError('gate_probe_num should be equal to up_probe_num for now')
        
        probe_out = self.act_fn(self.gate_proj(probe, cal_mlp_probe_out_dim_metric=True)) * self.up_proj(probe, cal_mlp_probe_out_dim_metric=True)

        # calculate score
        if 'calib' in cfg['prune_method'] or 'runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']:
            probe_out_dim_metric = self.pruning_module.cal_mlp_prune_metric(probe_out, self.down_proj.weight.data, cfg['prune_metric'], bsz_selected_indices, seq_selected_indices, global_metric_score_distribution=self.down_proj.get_global_metric_score_distribution(cur_batch_seq_len))
        else:
            probe_out_dim_metric = self.pruning_module.cal_mlp_prune_metric(probe_out, self.down_proj.weight.data, cfg['prune_metric'], bsz_selected_indices, seq_selected_indices)

        if 'flapratio' in cfg['prune_method']:
            probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.down_proj.pruning_ratio)
        else:
            probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, cfg['tc_multiple'])

        # extract matrix
        self.gate_proj.prepare_async_weight(out_dim_indices=probe_out_dim_indices)
        self.up_proj.prepare_async_weight(out_dim_indices=probe_out_dim_indices)
        self.down_proj.prepare_async_weight(in_dim_indices=probe_out_dim_indices)
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
            if cfg['calibration_stage'] == True:
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                return down_proj
            elif cfg['calibration_stage'] == False:
                if ('down_proj' in cfg['cust_tgt_modules'] or 'up_proj' in cfg['cust_tgt_modules'] or 'gate_proj' in cfg['cust_tgt_modules']) and self.layer_order not in cfg['skip_layers']:
                    if 'probe' in cfg['prune_method']:
                        probe_out_dim_indices = None
                        if cfg['mode'] == 'sync':
                            probe_out_dim_indices, probe_out = self.probe_process(x, **kwargs)
                            # count flops for probe
                            if cfg['onlyprobe'] == True:
                                # match the shape, and will not count the flops for this part
                                down_proj = torch.zeros((cfg['batch_size'], x.shape[1], self.hidden_size), device=x.device, dtype=x.dtype)
                                return down_proj
                        elif cfg['mode'] == 'asyncintra':
                            if 'post_layernorm_attn_residual' in kwargs:
                                _, _ = self.probe_process(kwargs['post_layernorm_attn_residual'], **kwargs)
                                # return
                            else:
                                # whole_batch_inference
                                pass
                                # raise ValueError('Invalid input for asyncintra mode')
                                # pass
                        if 'recordcommonchannel' in cfg['prune_method']:
                            self.mlp_cur_select_indices = probe_out_dim_indices.tolist()
                        # if 'recorddiff' in cfg['prune_method']:
                        #     out_dim_metric = self.pruning_module.cal_mlp_calib_prune_metric(self.down_proj.get_global_metric_score_distribution(), self.down_proj.weight.data, cfg['prune_metric'])
                        #     out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'])

                        #     tensor_A = probe_out_dim_indices.detach().view(-1).cpu().numpy()
                        #     tensor_B = out_dim_indices.detach().view(-1).cpu().numpy()
                        #     tensor_C = np.intersect1d(tensor_A, tensor_B)
                        #     self.diff_ratio = 1 - tensor_C.shape[0] / tensor_A.shape[0]

                        # print('mlp layer', self.layer_order, flush=True)
                        down_proj = self.down_proj(self.act_fn(self.gate_proj(x, out_dim_indices=probe_out_dim_indices)) * self.up_proj(x, out_dim_indices=probe_out_dim_indices), in_dim_indices=probe_out_dim_indices)
     
                        # if cfg['mode'] == 'asyncinter':
                        #     with torch.cuda.stream(cfg['cuda_stream1']):
                        #         _, _ = self.probe_process(x)
                        return down_proj
                    elif 'calib' in cfg['prune_method'] and ('runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']):

                        if cfg['mode'] == 'asyncinter':
                            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

                            # with torch.cuda.stream(cfg['cuda_stream1']):
                            if torch.all(self.down_proj.get_global_metric_score_distribution() == 0):
                                out_dim_indices = torch.arange(self.intermediate_size, dtype=torch.long).to(device=x.device)
                            else:
                                out_dim_metric = self.pruning_module.cal_mlp_calib_prune_metric(self.down_proj.get_global_metric_score_distribution(), self.down_proj.weight.data, cfg['prune_metric'])

                                if 'flapratio' in cfg['prune_method']:
                                    out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.down_proj.pruning_ratio)
                                else:
                                    out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'])

                            self.gate_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                            self.up_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                            self.down_proj.prepare_async_weight(in_dim_indices=out_dim_indices)
                        else:
                            raise ValueError('Invalid mode')
                        return down_proj
                    # only calib (baselines)
                    elif 'calib' in cfg['prune_method'] or 'flap' in cfg['prune_method'] or 'wandasp' in cfg['prune_method']:
                        # no ema or runningmean
                        if cfg['mode'] == 'asyncinter':
                            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

                            if cfg['cur_batch_index'] == 0:
                                if torch.all(self.down_proj.get_global_metric_score_distribution() == 0):
                                    out_dim_indices = torch.arange(self.intermediate_size, dtype=torch.long).to(device=x.device)
                                else:
                                    out_dim_metric = self.pruning_module.cal_mlp_calib_prune_metric(self.down_proj.get_global_metric_score_distribution(), self.down_proj.weight.data, cfg['prune_metric'])

                                    if 'flapratio' in cfg['prune_method'] or 'gridratio' in cfg['prune_method']:
                                        out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.down_proj.pruning_ratio)
                                    else:
                                        out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'])

                                self.gate_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                                self.up_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                                self.down_proj.prepare_async_weight(in_dim_indices=out_dim_indices)
                                if 'recordcommonchannel' in cfg['prune_method']:
                                    self.mlp_cur_select_indices = out_dim_indices.tolist()
                        else:
                            raise ValueError('Invalid mode')
                        return down_proj
                else:
                    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                    return down_proj

def repeat_kv(hidden_states: torch.Tensor, n_rep: int, heads_to_preserve=None) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    if heads_to_preserve is not None:
        heads_to_preserve = heads_to_preserve.to(hidden_states.device)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)[:, heads_to_preserve, :, :]
    else:
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
        self.pruning_module = HiddenRepresentationPruning(cfg, f'llama_attention_{layer_order}', config)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.layer_order = layer_order
        self.q_num_heads, self.k_num_heads, self.v_num_heads = self.num_heads, self.num_key_value_heads, self.num_key_value_heads
        self.heads_to_preserve = torch.arange(self.num_heads)
        self._init_rope()

    def is_GQA(self):
        return self.num_key_value_groups != 1
    
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.num_heads,
                self.head_dim,
                max_position_embeddings=cfg['max_seq_len'],
                base=self.rope_theta,
            )
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

        qk_prune_way = cfg['qk_prune_way']
        vo_prune_way = cfg['vo_prune_way']
        cur_batch_seq_len = hidden_states.size(1)
        # generate probe: rank
        if cfg['q_probe_ratio'] == cfg['k_probe_ratio'] and cfg['q_probe_ratio'] == cfg['v_probe_ratio']:
            if 'respick' in cfg['prune_method']:
                residual_for_probe = kwargs['respick']
            else:
                residual_for_probe = None
            probe, bsz_selected_indices, seq_selected_indices = generate_probe(hidden_states, cfg[f'q_probe_ratio'], residual_for_probe)
        else:
            raise ValueError('q_probe_num should be equal to k_probe_num and v_probe_num for now')

        bsz, q_len, _ = probe.size()
        self.q_num_heads, self.k_num_heads, self.v_num_heads = self.num_heads, self.num_key_value_heads, self.num_key_value_heads

        # copy orignal code and modify a little bit for probe pruning
        # currently does not implement for group attention, but it should work too
        query_states = self.q_proj(probe, cal_attn_probe_out_dim_metric=True)   
        key_states = self.k_proj(probe, cal_attn_probe_out_dim_metric=True)
        value_states = self.v_proj(probe, cal_attn_probe_out_dim_metric=True)

        query_states = query_states.view(query_states.shape[0], q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(key_states.shape[0], q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(value_states.shape[0], q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=cur_batch_seq_len)

        if q_len != cur_batch_seq_len:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids[:, seq_selected_indices])
            # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids[:, :q_len])
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (key_states.shape[0], self.q_num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(key_states.shape[0], self.q_num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if bsz_selected_indices is None:
            probe_attn_mask = attention_mask[:, :, :q_len, :kv_seq_len]
        else:
            probe_attn_mask = attention_mask[bsz_selected_indices, :, :q_len, :kv_seq_len]

        if probe_attn_mask is not None:
            if probe_attn_mask.size() != (key_states.shape[0], 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(key_states.shape[0], 1, q_len, kv_seq_len)}, but is {probe_attn_mask.size()}"
                )
            attn_weights = attn_weights + probe_attn_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # summed_attn_weights = attn_weights.sum(dim=(1, 2))
        # print('probe attn_weights', attn_weights, check_nan_inf(attn_weights), flush=True)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (key_states.shape[0], self.q_num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(key_states.shape[0], self.q_num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(key_states.shape[0], q_len, self.hidden_size)
        
        if 'calib' in cfg['prune_method'] or 'runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']:
            probe_out_dim_metric = self.pruning_module.cal_attn_prune_metric(attn_output, self.o_proj.weight.data, cfg['prune_metric'], bsz_selected_indices, seq_selected_indices, global_metric_score_distribution=self.o_proj.get_global_metric_score_distribution(cur_batch_seq_len))
        else:
            probe_out_dim_metric = self.pruning_module.cal_attn_prune_metric(attn_output, self.o_proj.weight.data, cfg['prune_metric'], bsz_selected_indices, seq_selected_indices)

        if 'flapratio' in cfg['prune_method']:
            probe_vo_out_dim_indices, self.remain_num_heads, self.heads_to_preserve = self.pruning_module.sort_attn_metric(probe_out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'], pruning_ratio=self.o_proj.pruning_ratio)
        else:
            probe_vo_out_dim_indices, self.remain_num_heads, self.heads_to_preserve = self.pruning_module.sort_attn_metric(probe_out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])

        
        if self.is_GQA():
            self.q_num_heads = self.remain_num_heads
            probe_qk_out_dim_indices = probe_vo_out_dim_indices
        else:
            self.q_num_heads, self.k_num_heads, self.v_num_heads = self.remain_num_heads, self.remain_num_heads, self.remain_num_heads
            probe_qk_out_dim_indices = probe_vo_out_dim_indices

        self.q_proj.prepare_async_weight(out_dim_indices=probe_qk_out_dim_indices)
        self.o_proj.prepare_async_weight(in_dim_indices=probe_vo_out_dim_indices)

        # MHA, for GQA, skip prune for simplicity, hard to get 4 consecutive heads to be pruned
        if not self.is_GQA():
            self.k_proj.prepare_async_weight(out_dim_indices=probe_qk_out_dim_indices)
            self.v_proj.prepare_async_weight(out_dim_indices=probe_vo_out_dim_indices)
        return probe_qk_out_dim_indices, probe_vo_out_dim_indices, attn_weights, attn_output, past_key_value


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
        # full inference
        bsz, q_len, _ = hidden_states.size()

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
            # print('torch.tensor(torch.finfo(attn_weights.dtype).min', torch.tensor(torch.finfo(attn_weights.dtype).min))
        

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # if 'mask' in cfg['prune_method']:
            # print('attn_weights before', attn_weights[0][0], attention_mask[0][0])
        # attn_weights = attn_weights + cfg['padding_mask']
        
        # value_states: bsz, self.num_key_value_heads, q_len, self.head_dim
        # value_states: 1, self.num_key_value_heads, q_len, self.head_dim
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        if cfg['pad_tokens'] is not None:
            cfg['pad_tokens'].to(attn_weights.device) 
            attn_output[cfg['pad_tokens']] = 0

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

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
        if cfg['calibration_stage'] == True:
            return self.attention_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
        elif cfg['calibration_stage'] == False :
            if ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']) and self.layer_order not in cfg['skip_layers']:
                bsz, q_len, _ = hidden_states.size()
                probe_qk_out_dim_indices, probe_vo_out_dim_indices = None, None
                if 'probe' in cfg['prune_method']:
                    qk_prune_way = cfg['qk_prune_way']
                    vo_prune_way = cfg['vo_prune_way']
                    if cfg['mode'] == 'sync':
                        probe_qk_out_dim_indices, probe_vo_out_dim_indices, attn_weights, attn_output, past_key_value = self.probe_process(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
                        # calculate probe's FLOPs
                        if cfg['onlyprobe'] == True:
                            attn_output = torch.zeros((cfg['batch_size'], hidden_states.shape[1], self.hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
                            if not output_attentions:
                                attn_weights = None
                            return attn_output, attn_weights, past_key_value
                    elif cfg['mode'] == 'asyncintra':
                        # if not, do full inference
                        if 'input_layernorm_mlp_residual' in kwargs:
                            _, _, _, _, _ = self.probe_process(kwargs['input_layernorm_mlp_residual'], attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
                            # return
                        else:
                            # full batch inference
                            pass
                            # raise ValueError('Invalid input for asyncintra mode')
                    
                    if 'recordcommonchannel' in cfg['prune_method']:
                        self.attn_cur_select_indices = probe_vo_out_dim_indices.tolist()
                    # if 'recorddiff' in cfg['prune_method']:
                    #     out_dim_metric = self.pruning_module.cal_attn_calib_prune_metric(self.o_proj.get_global_metric_score_distribution(), self.o_proj.weight.data, cfg['prune_metric'])
                    #     vo_out_dim_indices, self.remain_num_heads, self.heads_to_preserve = self.pruning_module.sort_attn_metric(out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])

                    #     tensor_A = probe_vo_out_dim_indices.detach().view(-1).cpu().numpy()
                    #     tensor_B = vo_out_dim_indices.detach().view(-1).cpu().numpy()
                    #     tensor_C = np.intersect1d(tensor_A, tensor_B)
                    #     self.diff_ratio = 1 - tensor_C.shape[0] / tensor_A.shape[0]


                    # --------------------------------------
                    #full inference
                    bsz, q_len, _ = hidden_states.size()
                    query_states = self.q_proj(hidden_states, out_dim_indices=probe_qk_out_dim_indices)
                    key_states = self.k_proj(hidden_states, out_dim_indices=probe_qk_out_dim_indices)
                    value_states = self.v_proj(hidden_states, out_dim_indices=probe_vo_out_dim_indices)

                    query_states = query_states.view(bsz, q_len, self.q_num_heads, self.head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, self.k_num_heads, self.head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, self.v_num_heads, self.head_dim).transpose(1, 2)

                    kv_seq_len = key_states.shape[-2]
                    if past_key_value is not None:
                        kv_seq_len += past_key_value[0].shape[-2]
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

                    if past_key_value is not None:
                        key_states = torch.cat([past_key_value[0], key_states], dim=2)
                        value_states = torch.cat([past_key_value[1], value_states], dim=2)

                    past_key_value = (key_states, value_states) if use_cache else None
                    if self.is_GQA():
                        key_states = repeat_kv(key_states, self.num_key_value_groups, self.heads_to_preserve)
                        value_states = repeat_kv(value_states, self.num_key_value_groups, self.heads_to_preserve)
                    else:
                        key_states = repeat_kv(key_states, self.num_key_value_groups)
                        value_states = repeat_kv(value_states, self.num_key_value_groups)

                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                    if attn_weights.size() != (bsz, self.q_num_heads, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention weights should be of size {(bsz, self.q_num_heads, q_len, kv_seq_len)}, but is"
                            f" {attn_weights.size()}"
                        )

                    if attention_mask is not None:
                        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                            raise ValueError(
                                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                            )
                        attn_weights = attn_weights + attention_mask
                        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
                        
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                    attn_output = torch.matmul(attn_weights, value_states)
                    if attn_output.size() != (bsz, self.q_num_heads, q_len, self.head_dim):
                        raise ValueError(
                            f"`attn_output` should be of size {(bsz, self.q_num_heads, q_len, self.head_dim)}, but is"
                            f" {attn_output.size()}"
                        )
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, self.q_num_heads * self.head_dim)
                    if cfg['pad_tokens'] is not None:
                        cfg['pad_tokens'].to(attn_weights.device) 
                        attn_output[cfg['pad_tokens']] = 0

                    attn_output = self.o_proj(attn_output, in_dim_indices=probe_vo_out_dim_indices)
                            
                    if not output_attentions:
                        attn_weights = None
                    
                    return attn_output, attn_weights, past_key_value
                elif 'calib' in cfg['prune_method'] and ('runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']):
                    time_start = time.time()
                    bsz, q_len, _ = hidden_states.size()

                    qk_prune_way = cfg['qk_prune_way']
                    vo_prune_way = cfg['vo_prune_way']
                    
                    query_states = self.q_proj(hidden_states)
                    key_states = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)

                    query_states = query_states.view(bsz, q_len, self.q_num_heads, self.head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, self.k_num_heads, self.head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, self.v_num_heads, self.head_dim).transpose(1, 2)

                    kv_seq_len = key_states.shape[-2]
                    if past_key_value is not None:
                        kv_seq_len += past_key_value[0].shape[-2]
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

                    if past_key_value is not None:
                        key_states = torch.cat([past_key_value[0], key_states], dim=2)
                        value_states = torch.cat([past_key_value[1], value_states], dim=2)

                    past_key_value = (key_states, value_states) if use_cache else None
                    if self.is_GQA():
                        key_states = repeat_kv(key_states, self.num_key_value_groups, self.heads_to_preserve)
                        value_states = repeat_kv(value_states, self.num_key_value_groups, self.heads_to_preserve)
                    else:
                        key_states = repeat_kv(key_states, self.num_key_value_groups)
                        value_states = repeat_kv(value_states, self.num_key_value_groups)

                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                    if attn_weights.size() != (bsz, self.q_num_heads, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention weights should be of size {(bsz, self.q_num_heads, q_len, kv_seq_len)}, but is"
                            f" {attn_weights.size()}"
                        )

                    if attention_mask is not None:
                        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                            raise ValueError(
                                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                            )
                        attn_weights = attn_weights + attention_mask
                        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                    attn_output = torch.matmul(attn_weights, value_states)
                    if attn_output.size() != (bsz, self.q_num_heads, q_len, self.head_dim):
                        raise ValueError(
                            f"`attn_output` should be of size {(bsz, self.q_num_heads, q_len, self.head_dim)}, but is"
                            f" {attn_output.size()}"
                        )
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, self.q_num_heads * self.head_dim)

                    attn_output = self.o_proj(attn_output)
                            
                    if not output_attentions:
                        attn_weights = None                      
                    
                    if cfg['mode'] == 'asyncinter':
                        # with torch.cuda.stream(cfg['cuda_stream1']):
                        if torch.all(self.o_proj.get_global_metric_score_distribution() == 0):
                            vo_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                        else:
                            out_dim_metric = self.pruning_module.cal_attn_calib_prune_metric(self.o_proj.get_global_metric_score_distribution(), self.o_proj.weight.data, cfg['prune_metric'])
                            if 'flapratio' in cfg['prune_method']:
                                vo_out_dim_indices, self.remain_num_heads, self.heads_to_preserve = self.pruning_module.sort_attn_metric(out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'], pruning_ratio=self.o_proj.pruning_ratio)
                            else:
                                vo_out_dim_indices, self.remain_num_heads, self.heads_to_preserve = self.pruning_module.sort_attn_metric(out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])

                        if self.is_GQA():
                            self.q_num_heads = self.remain_num_heads
                            qk_out_dim_indices = vo_out_dim_indices
                        else:
                            self.q_num_heads, self.k_num_heads, self.v_num_heads = self.remain_num_heads, self.remain_num_heads, self.remain_num_heads
                            qk_out_dim_indices = vo_out_dim_indices
                                
                        self.q_proj.prepare_async_weight(out_dim_indices=qk_out_dim_indices)
                        self.o_proj.prepare_async_weight(in_dim_indices=vo_out_dim_indices)
                        # MHA, for GQA, skip prune for simplicity
                        if not self.is_GQA():
                            self.k_proj.prepare_async_weight(out_dim_indices=qk_out_dim_indices)
                            self.v_proj.prepare_async_weight(out_dim_indices=vo_out_dim_indices)
                    else:
                        raise ValueError('please use asyncinter for calib+ema')
                    return attn_output, attn_weights, past_key_value
                elif 'calib' in cfg['prune_method'] or 'flap' in cfg['prune_method'] or 'wandasp' in cfg['prune_method']:

                    if cfg['mode'] == 'asyncinter':
                        time_start = time.time()
                        bsz, q_len, _ = hidden_states.size()

                        qk_prune_way = cfg['qk_prune_way']
                        vo_prune_way = cfg['vo_prune_way']
                        
                        query_states = self.q_proj(hidden_states)
                        key_states = self.k_proj(hidden_states)
                        value_states = self.v_proj(hidden_states)

                        query_states = query_states.view(bsz, q_len, self.q_num_heads, self.head_dim).transpose(1, 2)
                        key_states = key_states.view(bsz, q_len, self.k_num_heads, self.head_dim).transpose(1, 2)
                        value_states = value_states.view(bsz, q_len, self.v_num_heads, self.head_dim).transpose(1, 2)

                        kv_seq_len = key_states.shape[-2]
                        if past_key_value is not None:
                            kv_seq_len += past_key_value[0].shape[-2]
                        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

                        if past_key_value is not None:
                            key_states = torch.cat([past_key_value[0], key_states], dim=2)
                            value_states = torch.cat([past_key_value[1], value_states], dim=2)

                        past_key_value = (key_states, value_states) if use_cache else None
                        if self.is_GQA():
                            key_states = repeat_kv(key_states, self.num_key_value_groups, self.heads_to_preserve)
                            value_states = repeat_kv(value_states, self.num_key_value_groups, self.heads_to_preserve)
                        else:
                            key_states = repeat_kv(key_states, self.num_key_value_groups)
                            value_states = repeat_kv(value_states, self.num_key_value_groups)

                        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                        if attn_weights.size() != (bsz, self.q_num_heads, q_len, kv_seq_len):
                            raise ValueError(
                                f"Attention weights should be of size {(bsz, self.q_num_heads, q_len, kv_seq_len)}, but is"
                                f" {attn_weights.size()}"
                            )

                        if attention_mask is not None:
                            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                                raise ValueError(
                                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                                )
                            attn_weights = attn_weights + attention_mask
                            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

                        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        attn_output = torch.matmul(attn_weights, value_states)
                        if attn_output.size() != (bsz, self.q_num_heads, q_len, self.head_dim):
                            raise ValueError(
                                f"`attn_output` should be of size {(bsz, self.q_num_heads, q_len, self.head_dim)}, but is"
                                f" {attn_output.size()}"
                            )
                        attn_output = attn_output.transpose(1, 2).contiguous()
                        attn_output = attn_output.reshape(bsz, q_len, self.q_num_heads * self.head_dim)

                        attn_output = self.o_proj(attn_output)
                                
                        if not output_attentions:
                            attn_weights = None    

                        if cfg['cur_batch_index'] == 0:
                            if torch.all(self.o_proj.get_global_metric_score_distribution() == 0):
                                vo_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                            else:
                                # TODO: deal with rope
                                out_dim_metric = self.pruning_module.cal_attn_calib_prune_metric(self.o_proj.get_global_metric_score_distribution(), self.o_proj.weight.data, cfg['prune_metric'])
                                if 'flapratio' in cfg['prune_method']:
                                    vo_out_dim_indices, self.remain_num_heads, self.heads_to_preserve = self.pruning_module.sort_attn_metric(out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'], pruning_ratio=self.o_proj.pruning_ratio)
                                else:
                                    vo_out_dim_indices, self.remain_num_heads, self.heads_to_preserve = self.pruning_module.sort_attn_metric(out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])

                            if self.is_GQA():
                                self.q_num_heads = self.remain_num_heads
                                qk_out_dim_indices = vo_out_dim_indices
                            else:
                                self.q_num_heads, self.k_num_heads, self.v_num_heads = self.remain_num_heads, self.remain_num_heads, self.remain_num_heads
                                qk_out_dim_indices = vo_out_dim_indices
                                    
                            self.q_proj.prepare_async_weight(out_dim_indices=qk_out_dim_indices)
                            self.o_proj.prepare_async_weight(in_dim_indices=vo_out_dim_indices)
                            if 'recordcommonchannel' in cfg['prune_method']:
                                self.attn_cur_select_indices = qk_out_dim_indices.tolist()
                            # MHA, for GQA, skip prune for simplicity
                            if not self.is_GQA():
                                self.k_proj.prepare_async_weight(out_dim_indices=qk_out_dim_indices)
                                self.v_proj.prepare_async_weight(out_dim_indices=vo_out_dim_indices)
                        return attn_output, attn_weights, past_key_value
                    else:
                        raise ValueError('Invalid mode')
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
        self.llamaconfig = config

        # if self.layer_order not in cfg['skip_layers']:
        self.cur_attn_inference_duration = 0
        self.cur_mlp_inference_duration = 0

        self.is_changing_gpu = False
        self.whole_batch_calculation = torch.cuda.Event(enable_timing=False, blocking=False)

        self.start_event = torch.cuda.Event(enable_timing=True, blocking=False)
        self.end_event = torch.cuda.Event(enable_timing=True, blocking=False)

    def check_asyncintra_for_next_mlp(self):
        if ('down_proj' in cfg['cust_tgt_modules'] or 'up_proj' in cfg['cust_tgt_modules'] or 'gate_proj' in cfg['cust_tgt_modules']) \
            and self.layer_order not in cfg['skip_layers'] \
            and cfg['calibration_stage'] == False \
            and cfg['mode'] == 'asyncintra' \
            and 'probe' in cfg['prune_method']:
            return True
        return False
    
    # def check_asyncintra_for_next_attention(self, **kwargs):
    #     # TODO: hardcode skip layers
    #     if ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']) \
    #         and self.layer_order not in cfg['skip_layers'][:-1] \
    #         and cfg['calibration_stage'] == False \
    #         and cfg['mode'] == 'asyncintra' \
    #         and 'probe' in cfg['prune_method']:
            
    #         # if kwargs['next_layer'] != 'changing_gpu' and kwargs['next_layer'] != 'last_layer':
    #         if kwargs['next_layer'] != None:
    #             return True
            
    #     return False
    
    def check_asyncintra_for_next_attention(self, **kwargs):
        # TODO: hardcode skip layers
        if ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']) \
            and self.layer_order not in cfg['skip_layers'] \
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
        # print('self.layer_order', self.layer_order, self.mlp.gate_proj.weight.device, self.mlp.gate_proj.weight.device.index, flush=True)
        # residual = hidden_states
        # self.whole_batch_calculation.record()   
        
        
        # cur_device = self.mlp.gate_proj.weight.device

        # print(torch.cuda.current_stream(cur_device), torch.cuda.current_stream(cur_device).device)
        # cur_custom_stream = cfg['custom_cuda_streams'][cur_gpu_index]
        # print('cur_custom_stream', cur_custom_stream.device, id(cur_custom_stream), flush=True)

        # cur_default_stream = cfg['default_cuda_streams'][cur_gpu_index]
        # print('cur_default_stream', cur_default_stream.device, id(cur_default_stream), flush=True)
        # def probe_mlp_inf(residual):
        #     if self.check_asyncintra_for_next_mlp():
        #         # with torch.cuda.stream(cur_custom_stream):
        #         # cur_custom_stream.wait_event(self.whole_batch_calculation)  
        #         print('cur_custom_stream in thread', cur_custom_stream.device, flush=True) 
        #         try:
        #             post_layernorm_attn_residual = self.post_attention_layernorm(residual)
        #             self.mlp(hidden_states, post_layernorm_attn_residual=post_layernorm_attn_residual, respick=residual)
        #         except Exception as e:
        #             print(f"Error in probe_mlp_inf: {e}")
        #             traceback.print_exc() 


        # def full_attn_inf(hidden_states, residual):
        #     # with torch.cuda.stream(cur_default_stream):
        #     if self.layer_order in cfg['gpu_breakpoints'] and cfg['mode'] == 'asyncintra':
        #         print('breakpoint', self.layer_order, flush=True)
        #         self.self_attn.q_proj.prepare_async_weight()
        #         self.self_attn.o_proj.prepare_async_weight()
        #         self.self_attn.k_proj.prepare_async_weight()
        #         self.self_attn.v_proj.prepare_async_weight()
        #     try:
        #         hidden_states = self.input_layernorm(hidden_states)
        #         hidden_states, self_attn_weights, present_key_value = self.self_attn(
        #             hidden_states=hidden_states,
        #             attention_mask=attention_mask,
        #             position_ids=position_ids,
        #             past_key_value=past_key_value,
        #             output_attentions=output_attentions,
        #             use_cache=use_cache,
        #             respick=residual
        #         )
        #         hidden_states = residual + hidden_states
        #         return hidden_states
        #     except Exception as e:
        #         print(f"Error in full_attn_inf: {e}")
        #         traceback.print_exc() 

        

        # probe_mlp_inf(residual)
        # hidden_states = full_attn_inf(hidden_states, residual)
        
    

        
        # # if self.check_asyncintra_for_next_attention(**kwargs):
        # # torch.cuda.synchronize()


        # residual = hidden_states
        # self.whole_batch_calculation.record()


        # def probe_attn_inf(residual):
        #     if self.check_asyncintra_for_next_attention(**kwargs):
        #         # with torch.cuda.stream(cur_custom_stream):
        #             # cur_custom_stream.wait_event(self.whole_batch_calculation)
        #         try:
        #             # cfg[f'cuda_events_mlp_{self.layer_order}'].wait(cfg['cuda_stream1'])
        #             input_layernorm_mlp_residual = kwargs['next_layer'].input_layernorm(residual)
        #             kwargs['next_layer'].self_attn(
        #                 hidden_states=hidden_states,
        #                 attention_mask=attention_mask,
        #                 position_ids=position_ids,
        #                 past_key_value=past_key_value,
        #                 output_attentions=output_attentions,
        #                 use_cache=use_cache,
        #                 input_layernorm_mlp_residual=input_layernorm_mlp_residual,
        #                 respick=residual
        #             )
        #         except Exception as e:
        #             print(f"Error in probe_attn_inf: {e}")
        #             traceback.print_exc() 
        #     else:
        #         # break point
        #         pass
        
        # def full_mlp_inf(hidden_states, residual):
        #     # with torch.cuda.stream(cur_default_stream):
        #     try:
        #         hidden_states = self.post_attention_layernorm(hidden_states)
        #         hidden_states = self.mlp(hidden_states, respick=residual)
        #         return hidden_states
        #     except Exception as e:
        #         print(f"Error in full_mlp_inf: {e}")
        #         traceback.print_exc() 
    
        

        # probe_attn_inf(residual)
        # hidden_states = full_mlp_inf(hidden_states, residual)
        # hidden_states = residual + hidden_states

        

        # ---
        # print('self.layer_order', self.layer_order, self.mlp.gate_proj.weight.device, self.mlp.gate_proj.weight.device.index, flush=True)
        # residual = hidden_states
        # self.whole_batch_calculation.record()   
        
        
        # cur_gpu_index = self.mlp.gate_proj.weight.device.index
        # cur_custom_stream = cfg['custom_cuda_streams'][cur_gpu_index]
        # print('cur_custom_stream', cur_custom_stream.device, id(cur_custom_stream), flush=True)

        # cur_default_stream = cfg['default_cuda_streams'][cur_gpu_index]
        # print('cur_default_stream', cur_default_stream.device, id(cur_default_stream), flush=True)
        # def full_attn_inf(result_dict, hidden_states, residual, cur_default_stream):
        #     # with torch.cuda.stream(cur_default_stream):
        #     if self.layer_order in cfg['gpu_breakpoints'] and cfg['mode'] == 'asyncintra':
        #         print('breakpoint', self.layer_order, flush=True)
        #         full_indices = torch.arange(self.hidden_size, dtype=torch.int32)
        #         self.self_attn.q_proj.prepare_async_weight(out_dim_indices=full_indices)
        #         self.self_attn.o_proj.prepare_async_weight(in_dim_indices=full_indices)
        #         self.self_attn.k_proj.prepare_async_weight(out_dim_indices=full_indices)
        #         self.self_attn.v_proj.prepare_async_weight(out_dim_indices=full_indices)
        #     try:
        #         hidden_states = self.input_layernorm(hidden_states)
        #         hidden_states, self_attn_weights, present_key_value = self.self_attn(
        #             hidden_states=hidden_states,
        #             attention_mask=attention_mask,
        #             position_ids=position_ids,
        #             past_key_value=past_key_value,
        #             output_attentions=output_attentions,
        #             use_cache=use_cache,
        #             respick=residual
        #         )
        #         hidden_states = residual + hidden_states
        #         result_dict['hidden_states'] = hidden_states
        #     except Exception as e:
        #         print(f"Error in full_attn_inf: {e}")
        #         traceback.print_exc() 

        # def probe_mlp_inf(result_dict, residual, cur_custom_stream):
        #     if self.check_asyncintra_for_next_mlp():
        #         with torch.cuda.stream(cur_custom_stream):
        #             cur_custom_stream.wait_event(self.whole_batch_calculation)  
        #             print('cur_custom_stream in thread', cur_custom_stream.device, flush=True) 
        #             try:
        #                 post_layernorm_attn_residual = self.post_attention_layernorm(residual)
        #                 if 'resinfo' in cfg['prune_method']: 
        #                     result_dict['post_layernorm_attn_residual'] = post_layernorm_attn_residual
        #                 self.mlp(hidden_states, post_layernorm_attn_residual=post_layernorm_attn_residual, respick=residual)
        #             except Exception as e:
        #                 print(f"Error in probe_mlp_inf: {e}")
        #                 traceback.print_exc() 

        # # Create threads
        # full_attn_inf_result = {}
        # probe_mlp_inf_result = {}
        # thread1 = threading.Thread(target=full_attn_inf, args=(full_attn_inf_result, hidden_states, residual, cur_default_stream))
        # thread2 = threading.Thread(target=probe_mlp_inf, args=(probe_mlp_inf_result, residual, cur_custom_stream))

        # # Start the threads
        # thread1.start()
        # thread2.start()
        # thread1.join()
        # thread2.join()


        
        # # if self.check_asyncintra_for_next_attention(**kwargs):
        # # torch.cuda.synchronize()

        # hidden_states = full_attn_inf_result['hidden_states']
        # if 'resinfo' in cfg['prune_method']:
        #     self.attn_sign_match_percentage, self.attn_l1_diff_percentage, self.attn_cosine_similarity = cal_res_hidden_state_diff(hidden_states, residual)
        # residual = hidden_states
        # self.whole_batch_calculation.record()
        # # if cfg['change_back_to_asyncintra_on_diff_gpu'] == True:
        # #     cfg['mode'] = 'asyncintra'
        # #     cfg['change_back_to_asyncintra_on_diff_gpu'] = False

        
            

        # def full_mlp_inf(result_dict, hidden_states, residual, cur_default_stream):
        #     # with torch.cuda.stream(cur_default_stream):
        #     try:
        #         hidden_states = self.post_attention_layernorm(hidden_states)
        #         hidden_states = self.mlp(hidden_states, respick=residual)
        #         result_dict['hidden_states'] = hidden_states
        #     except Exception as e:
        #         print(f"Error in full_mlp_inf: {e}")
        #         traceback.print_exc() 
    
        # def probe_attn_inf(result_dict, residual, cur_custom_stream):
        #     if self.check_asyncintra_for_next_attention(**kwargs):
        #         with torch.cuda.stream(cur_custom_stream):
        #             cur_custom_stream.wait_event(self.whole_batch_calculation)
        #             try:
        #                 # cfg[f'cuda_events_mlp_{self.layer_order}'].wait(cfg['cuda_stream1'])
        #                 input_layernorm_mlp_residual = kwargs['next_layer'].input_layernorm(residual)
        #                 kwargs['next_layer'].self_attn(
        #                     hidden_states=hidden_states,
        #                     attention_mask=attention_mask,
        #                     position_ids=position_ids,
        #                     past_key_value=past_key_value,
        #                     output_attentions=output_attentions,
        #                     use_cache=use_cache,
        #                     input_layernorm_mlp_residual=input_layernorm_mlp_residual,
        #                     respick=residual
        #                 )
        #                 if 'resinfo' in cfg['prune_method']: 
        #                     result_dict['input_layernorm_mlp_residual'] = input_layernorm_mlp_residual
                        
        #             except Exception as e:
        #                 print(f"Error in probe_attn_inf: {e}")
        #                 traceback.print_exc() 
        #     else:
        #         # break point
        #         pass

        
        # full_mlp_inf_result = {}
        # probe_attn_inf_result = {}
        # thread1 = threading.Thread(target=full_mlp_inf, args=(full_mlp_inf_result, hidden_states, residual, cur_default_stream))
        # thread2 = threading.Thread(target=probe_attn_inf, args=(probe_attn_inf_result, residual, cur_custom_stream))

        # # Start the threads
        # thread1.start()
        # thread2.start()

        # thread1.join()
        # thread2.join()

        # # if self.check_asyncintra_for_next_mlp():
        # #     torch.cuda.synchronize()
        # # torch.cuda.synchronize()
        # hidden_states = full_mlp_inf_result['hidden_states']
        # # print('hidden_states_shape', hidden_states.shape, hidden_states.device, flush=True)
        # # print('residual_shape', residual.shape, residual.device, flush=True)
        # hidden_states = residual + hidden_states
        
        # if 'resinfo' in cfg['prune_method']:
        #     self.mlp_sign_match_percentage, self.mlp_l1_diff_percentage, self.mlp_cosine_similarity = cal_res_hidden_state_diff(hidden_states, residual)
        #     print('self.layer_order', self.layer_order, flush=True)
        #     print('self.mlp_sign_match_percentage', self.mlp_sign_match_percentage, flush=True)
        #     print('self.mlp_l1_diff_percentage', self.mlp_l1_diff_percentage, flush=True)
        #     print('self.mlp_cosine_similarity', self.mlp_cosine_similarity, flush=True)


        
        residual = hidden_states
        # print('residual start new layer', self.layer_order, residual[0], check_nan_inf(residual), residual.shape, flush=True)


        torch.cuda.synchronize()
        self.start_event.record()
        hidden_states = self.input_layernorm(hidden_states)

        if self.check_asyncintra_for_next_attention():
            input_layernorm_mlp_residual = self.input_layernorm(kwargs['last_layer_residual'], keepdatatype=True)
            # input_layernorm_mlp_residual = kwargs['last_layer_residual']
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            # norm=residual,
            respick=kwargs['last_layer_residual'],
            input_layernorm_mlp_residual=input_layernorm_mlp_residual
        )
        else:
            input_layernorm_mlp_residual = None
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                # norm=residual,
                respick=residual,
            )
        
        # print('hidden state after attn', self.layer_order, hidden_states[0], check_nan_inf(hidden_states), flush=True)
        # print('residual after attn', self.layer_order, id(residual), residual[0], check_nan_inf(residual), flush=True)
        hidden_states = residual + hidden_states

        if self.check_asyncintra_for_next_mlp():
            post_layernorm_attn_residual = self.post_attention_layernorm(residual, keepdatatype=True)
            # post_layernorm_attn_residual = residual
            respick = residual
        else:
            post_layernorm_attn_residual = None

        residual = hidden_states
        # print('output after attn', self.layer_order, check_nan_inf(residual), residual[0], flush=True)
        self.end_event.record()
        torch.cuda.synchronize()
        if hasattr(self, 'cur_attn_inference_duration') and cfg['cur_batch_index'] >= 1:
            self.cur_attn_inference_duration += self.start_event.elapsed_time(self.end_event)

        self.start_event.record()
        start_time = time.time()
        hidden_states = self.post_attention_layernorm(hidden_states)
        # print('hidden_states after mlp input norm', self.layer_order, check_nan_inf(hidden_states), hidden_states[0], flush=True)
        if self.check_asyncintra_for_next_mlp():
            hidden_states = self.mlp(hidden_states, respick=respick, post_layernorm_attn_residual=post_layernorm_attn_residual)
        else:
            hidden_states = self.mlp(hidden_states, respick=residual)
        # print('hidden_states mlp output', self.layer_order, check_nan_inf(hidden_states), hidden_states[0], flush=True)
        hidden_states = residual + hidden_states


        self.end_event.record()
        torch.cuda.synchronize()
        if hasattr(self, 'cur_mlp_inference_duration') and cfg['cur_batch_index'] >= 1:
            # self.cur_mlp_inference_duration += time.time() - start_time
            self.cur_mlp_inference_duration += self.start_event.elapsed_time(self.end_event)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if 'asyncintra' in cfg['mode']:
            return outputs, residual
        else:
            return outputs, None


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

        # for pad token, set to 0
        inputs_embeds[attention_mask == 0] = 0
        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # padding_mask = attention_mask
            cfg['first_one_indices'] = torch.argmax((attention_mask == 1).int(), dim=1).unsqueeze_(1)
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )


        # embed positions
        hidden_states = inputs_embeds
        # hidden_states = torch.zeros_like(inputs_embeds)
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

        last_layer_residual = None
        # torch.cuda.nvtx.range_push("layer start".format(idx))
        for idx, decoder_layer in enumerate(self.layers):
            # hidden_states = hidden_states.to(decoder_layer.input_layernorm.weight.device)
            # print(idx, decoder_layer.input_layernorm.weight.device, hidden_states.device, flush=True)
            # print('decoder_layer.input_layernorm.weight.device', decoder_layer.input_layernorm.weight.device, flush=True)
            # print('decoder_layerqkvodevice', decoder_layer.self_attn.q_proj.weight.device, decoder_layer.self_attn.k_proj.weight.device, decoder_layer.self_attn.v_proj.weight.device, decoder_layer.self_attn.o_proj.weight.device, flush=True)
            # print('decoder_layermlpdevice', decoder_layer.mlp.down_proj.weight.device, decoder_layer.mlp.up_proj.weight.device, decoder_layer.mlp.gate_proj.weight.device, flush=True)
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
                
                start_time = time.time()

                next_layer = get_next_layer(self.layers, idx)

                layer_outputs, last_layer_residual = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    # next_layer=next_layer,
                    last_layer_residual=last_layer_residual
                )

                # print('layer_outputs', idx, layer_outputs[0])
                # Record the end time
                # stop_event.record()

                # Wait for all the operations to complete
                 # Wait for the events to be recorded!

                # Calculate the elapsed time
                # elapsed_time_ms = start_event.elapsed_time(stop_event)
        
               
                # print(f'layerdevice, {decoder_layer.mlp.gate_proj.weight.device}')
            

            # print('layer_outputs', layer_outputs, flush=True)
            hidden_states = layer_outputs[0]
            

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)


        duration = time.time() - start_time
        # torch.cuda.nvtx.range_pop()
        # print(f"layer: {idx}, Elapsed time: {duration} milliseconds")
        # print(f'layerdevice, {decoder_layer.mlp.gate_proj.weight.device}')
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
        
        loss = None
        if labels is not None:
            # occupy too many gpu memory when using llama-3-8b (large vocab size)
            # cant put in 1 a100-40gb
            if 'clm' in cfg['task_name'] and cfg['max_seq_len'] == 2048:
                original_device = logits.device
                logits = logits.cpu()
                logits = logits.float()
                labels = labels.cpu()

                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss.to(original_device)
            else:
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

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