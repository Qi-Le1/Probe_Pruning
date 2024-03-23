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
from ..pruning_module import HiddenRepresentationPruning, cal_intersection_ratio, cal_prune_metric, cal_calib_prune_metric
from module import nearest_even_number
from torch.nn.functional import cosine_similarity
from .utils import nml_process, max_process
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
        self.gate_proj.cal_total_flops = True
        self.up_proj.cal_total_flops = True
        self.down_proj.cal_total_flops = True

        self.act_fn = ACT2FN[config.hidden_act]

        self.layer_order = layer_order
        self.custom_duration = 0
        self.cal_total_flops = True
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
                        print('xshape', x.shape, flush=True)
                        if 'similarityprobe' in cfg['prune_method']:
                            start_time = time.time()
                            
                            if self.input_norm_gate_weight is None:
                                self.input_norm_gate_weight = torch.norm(self.gate_proj.weight.data, p=2, dim=0).reshape(1, 1, -1)
                            if self.input_norm_up_weight is None:
                                self.input_norm_up_weight = torch.norm(self.up_proj.weight.data, p=2, dim=0).reshape(1, 1, -1)

                            # def cal_sign_agreement_metrix(x_flattened):
                            #     strength = torch.norm(x_flattened, p=2, dim=0)
                            #     # Calculate the number of positions to select (top 10%)
                            #     top_k = max(int(0.03 * strength.numel()), 1)  # Ensure at least one position is selected
                            #     # print('top_k', top_k, strength.numel(), strength.shape, flush=True)
                            #     # Use torch.topk to find the top k positions. 
                            #     # torch.topk returns values and their corresponding indices.
                            #     top_values, top_indices = torch.topk(strength, k=top_k)

                            #     top_positions_flat = x_flattened[:, top_indices]  # [bsz, top_k]
                            #     print('top_positions_flat', top_positions_flat, flush=True)
                            #     signs_top_positions_flat = torch.sign(top_positions_flat)
                            #     # print('signs_top_positions_flat', signs_top_positions_flat, flush=True)
                            #     # sign_similarity = signs_top_positions_flat * signs_top_positions_flat.transpose(0, 1)
                            #     # Expand dimensions for broadcasting
                            #     expanded_signs = signs_top_positions_flat.unsqueeze(1)  # Shape: [bsz, 1, top_k]
                            #     # Repeat signs for comparison across all pairs
                            #     repeated_signs = signs_top_positions_flat.unsqueeze(0)  # Shape: [1, bsz, top_k]

                            #     # Element-wise multiplication to check sign agreement (-1 * -1 = 1, 1 * 1 = 1, else = -1 or 0)
                            #     sign_agreement = expanded_signs * repeated_signs  # Shape: [bsz, bsz, top_k]

                            #     # Sum over the top_k dimension to count the number of agreements per pair
                            #     sign_agreement_matrix = sign_agreement.sum(dim=-1)  # Shape: [bsz, bsz]
                            #     # print('sign_agreement_matrix v1', sign_agreement_matrix, flush=True)
                            #     sign_agreement_matrix = sign_agreement_matrix / top_k
                            #     torch.set_printoptions(threshold=5000)  # Adjust the number as needed
                            #     print('sign_agreement_matrix', sign_agreement_matrix, flush=True)
                            # #     print('top_positions_flat', top_positions_flat, flush=True)
                            # #     # Normalize
                            # #     norm_signs_top_positions_flat = signs_top_positions_flat / (torch.norm(signs_top_positions_flat, p=2, dim=-1, keepdim=True) + 1e-9)
                            # #    # Assuming norm_top_positions_flat is [bsz, top_k]
                            # #     similarity_matrix = torch.matmul(norm_signs_top_positions_flat, norm_signs_top_positions_flat.transpose(0, 1))
                            
                            # x_temp_gate = x * self.input_norm_gate_weight
                            # x_temp_up = x * self.input_norm_up_weight
                            # print('\nx_temp_gate')
                            # cal_sign_agreement_metrix(x_temp_gate.view(x_temp_gate.size(0), -1))
                            # print('\nx_temp_up')
                            # cal_sign_agreement_metrix(x_temp_up.view(x_temp_up.size(0), -1))
                            # x_flattened = x.view(x.size(0), -1) 
                            def cal_dot_product_matrix(x_flattened):
                                strength = torch.norm(x_flattened, p=2, dim=0)
                                # Calculate the number of positions to select (top 3% here as per your code)
                                top_k = max(int(0.03 * strength.numel()), 1)  # Ensure at least one position is selected
                                top_values, top_indices = torch.topk(strength, k=top_k)

                                top_positions_flat = x_flattened[:, top_indices]  # [bsz, top_k]
                                
                                # Calculate dot product matrix
                                # Normalize the vectors to only measure directionality
                                # norm_top_positions_flat = top_positions_flat / (torch.norm(top_positions_flat, p=2, dim=-1, keepdim=True) + 1e-9)
                                
                                # Dot product similarity (using matrix multiplication for efficiency)
                                # Here, we're effectively doing dot product because the vectors are normalized
                                dot_product_matrix = torch.matmul(top_positions_flat, top_positions_flat.transpose(0, 1))  # Shape: [bsz, bsz]
                                
                                # Optionally, normalize the dot product matrix to scale the values between -1 and 1
                                # This step may not be necessary since we're already working with normalized vectors
                                # dot_product_matrix = dot_product_matrix / top_k  # Normalize if needed
                                
                                torch.set_printoptions(threshold=5000)  # Adjust the number as needed
                                print('dot_product_matrix', dot_product_matrix, flush=True)

                            # Example of how to call the function
                            # Assuming x is your input tensor
                            x_temp_gate = x * self.input_norm_gate_weight
                            x_temp_up = x * self.input_norm_up_weight
                            # Flatten x as needed and pass to the function
                            cal_dot_product_matrix(x_temp_gate.view(x_temp_gate.size(0), -1))
                            cal_dot_product_matrix(x_temp_up.view(x_temp_up.size(0), -1))

                            end_time = time.time()
                            print('similarity_duration', self.layer_order, end_time - start_time, flush=True)

                            # similarity_matrix = similarity_matrix.mean(dim=-1)
                            # print('similarity_matrix', similarity_matrix, flush=True)

                            # gate_weight_flatten = self.gate_proj.weight.data.view(-1)
                            # down_weight_flatten = self.down_proj.weight.data.view(-1)
                            # up_weight_flatten = self.up_proj.weight.data.view(-1)

                            # top_k_gate_weight, top_k_gate_indices = torch.topk(gate_weight_flatten, k=top_k)
                            # top_k_down_weight, top_k_down_indices = torch.topk(down_weight_flatten, k=top_k)
                            # top_k_up_weight, top_k_up_indices = torch.topk(up_weight_flatten, k=top_k)
                            # print('top_k_gate_weight', top_k_gate_weight, top_k_gate_indices, flush=True)
                            # print('top_k_down_weight', top_k_down_weight, top_k_down_indices, flush=True)
                            # print('top_k_up_weight', top_k_up_weight, top_k_up_indices, flush=True)
                            abs_x = torch.abs(x).to(torch.float32)
                            sum_across_bsz = abs_x.sum(dim=0, keepdim=True)
                            # proportion = abs_x / torch.sum(abs_x, dim=0, keepdim=True)
                            proportion = (abs_x / (sum_across_bsz + 1e-10)).to(x.dtype)
                            # proportion = 10
                            # print('proportion ', proportion, flush=True)
                            comp_across_bsz = torch.sum(x * proportion, dim=0)
                            comp_across_bsz = comp_across_bsz.unsqueeze(0)
                        # if 'gauexp' in cfg['prune_method']:
                        #     # mean_for_all_batches = self.gate_proj.mean_for_all_batches
                        #     # std_for_all_batches = self.gate_proj.std_for_all_batches
                        #     mean_for_all_batches, std_for_all_batches = self.gate_proj.get_global_input_distribution()
                        #     torch.set_printoptions(threshold=2000)
                        #     # print('mean_for_all_batches', mean_for_all_batches, flush=True)
                        #     # print('std_for_all_batches', std_for_all_batches, flush=True)
                        #     # Compute z-scores
                        #     # print('x', x, flush=True)
                        #     # z_scores = (x - mean_for_all_batches) / std_for_all_batches
                        #     delta = x - mean_for_all_batches
                        #     weights = torch.abs(delta) ** 2
                        #     # print('z_scores', z_scores, flush=True)

                        #     # Calculate exponential weights
                        #     # weights = torch.exp(torch.abs(z_scores))
                        #     # print('weights', weights, flush=True)
                        #     weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-10)
                        #     # print('weights after nml', weights, flush=True)
                        #     # Apply weights and sum across the batch dimension
                        #     comp_across_bsz = torch.sum(weights * delta + mean_for_all_batches, dim=0)
                        # elif 'l2nml' in cfg['prune_method']:
                        #     abs_x = torch.abs(x).to(torch.float32)
                        #     norm_across_bsz = torch.norm(x, p=2, dim=0, keepdim=True)
                        #     proportion = (abs_x / (norm_across_bsz + 1e-10)).to(x.dtype)
                        #     comp_across_bsz = torch.sum(x * proportion, dim=0)
                        #     comp_across_bsz = comp_across_bsz.unsqueeze(0)
                        # elif 'deltaplusmeannml' in cfg['prune_method']:
                        #     mean_for_batch = x.mean(dim=0, keepdim=True)
                        #     abs_delta = torch.abs(x - mean_for_batch).to(torch.float32)
                        #     abs_sum = abs_delta.sum(dim=0, keepdim=True) + torch.abs(mean_for_batch)
                        #     porportion = (torch.abs(mean_for_batch) + abs_delta) / (abs_sum + 1e-10) / torch.sum((torch.abs(mean_for_batch) + abs_delta) / (abs_sum + 1e-10), dim=0, keepdim=True)
                        #     comp_across_bsz = (porportion * x).to(x.dtype)
                        #     comp_across_bsz = comp_across_bsz.unsqueeze(0)
                        # elif 'deltanml' in cfg['prune_method']:
                        #     # mean_for_all_batches, std_for_all_batches = self.gate_proj.get_global_input_distribution()
                        #     mean_for_batch = x.mean(dim=0, keepdim=True)
                        #     delta = (x - mean_for_batch).to(torch.float32)
                        #     abs_delta = torch.abs(delta)
                        #     # proportion = abs_x / torch.sum(abs_x, dim=0, keepdim=True)
                        #     proportion = (abs_delta / (abs_delta.sum(dim=0, keepdim=True) + 1e-10)).to(x.dtype)
                        #     # proportion = 10
                        #     # print('proportion ', proportion, flush=True)
                        #     comp_across_bsz = torch.sum(delta * proportion + mean_for_batch, dim=0).to(x.dtype)
                        #     comp_across_bsz = comp_across_bsz.unsqueeze(0)
                        elif 'avg' in cfg['prune_method']:
                            # abs_x = torch.abs(x).to(torch.float32)
                            # porportion = abs_x / abs_x.sum(dim=0, keepdim=True)
                            # print('porportion', porportion, porportion.dtype, porportion.shape, flush=True)
                            # comp_across_bsz = ((x.to(torch.float32) * porportion).sum(dim=0)).to(x.dtype)

                            # parts = torch.split(x, bsz / cfg['probe_num'])
                            comp_across_bsz = x.mean(axis=0)
                            comp_across_bsz = comp_across_bsz.unsqueeze(0)

                            
                        elif 'max' in cfg['prune_method']:
                            comp_across_bsz = max_process(x, cfg['probe_num'], cfg['probe_size'])
                        elif 'median' in cfg['prune_method']:
                            comp_across_bsz = median_process(x, cfg['probe_num'], cfg['probe_size'])
                        # elif 'std' in cfg['prune_method']:
                        #     abs_x = torch.abs(x)
                        #     # Calculate mean and std deviation across the batch dimension
                        #     mean_x = abs_x.mean(dim=0, keepdim=True)
                        #     std_x = abs_x.std(dim=0, keepdim=True)
                        #     # Normalize using mean and standard deviation
                        #     normalized_x = (abs_x - mean_x) / (std_x + 1e-6)
                        #     proportion = normalized_x / normalized_x.sum(dim=0, keepdim=True)
                        #     comp_across_bsz = (x * proportion).sum(dim=0)
                        #     comp_across_bsz = comp_across_bsz.unsqueeze(0)
                        # elif 'log' in cfg['prune_method']:
                        #     abs_x = torch.abs(x)
                        #     log_abs_x = torch.log1p(abs_x)  # log1p for log(x+1) to handle zeros
                        #     sum_log_abs_x = log_abs_x.sum(dim=0, keepdim=True)
                        #     proportion = log_abs_x / (sum_log_abs_x + 1e-6)
                        #     comp_across_bsz = (x * proportion).sum(dim=0)
                        #     comp_across_bsz = comp_across_bsz.unsqueeze(0)
                        elif 'fullinf' in cfg['prune_method']:
                            comp_across_bsz = x
                        # elif 'pcabszseq' in cfg['prune_method']:
                        #     start_time = time.time()
                        #     inp = x.reshape(-1, x.shape[-1]).to(torch.float32).t()
                        #     # This V is the transpose of the V in the SVD
                        #     U, S, V = torch.svd(inp)
                        #     extract_element = int(round(1/bsz * inp.shape[-1]))
                        #     print('extract_element', extract_element, flush=True)
                        #     comp_across_bsz = torch.matmul(inp, V.T[:, :extract_element]).to(x.dtype)
                        #     comp_across_bsz = comp_across_bsz.t()
                        #     end_time = time.time()
                        #     print('svd_duration', end_time - start_time, flush=True)
                        # elif 'twoprobe' in cfg['prune_method']:
                        #     probe_one = x.mean(axis=0)
                        #     probe_two = x.mean(axis=1)
                        #     comp_across_bsz = torch.cat((probe_one, probe_two), dim=0)
                        # elif 'normbsz' in cfg['prune_method']:
                        #     comp_across_bsz = torch.norm(x, p=2, dim=0)
                        #     comp_across_bsz = comp_across_bsz.unsqueeze(0)
                        else:
                            start_time = time.time()
                            if cfg['gate_probe_num'] == cfg['up_probe_num']:
                                comp_across_bsz_gate = nml_process(kwargs['post_attn_residual'], cfg['gate_probe_num'], cfg['gate_probe_size'])
                                comp_across_bsz_up = comp_across_bsz_gate
                            else:
                                comp_across_bsz_gate = nml_process(kwargs['post_attn_residual'], cfg['gate_probe_num'], cfg['gate_probe_size'])
                                comp_across_bsz_up = nml_process(kwargs['post_attn_residual'], cfg['up_probe_num'], cfg['up_probe_size'])
        
                            # comp_across_bsz = nml_process(x, cfg['probe_num'], cfg['probe_size'])
                            # abs_x = torch.clamp(torch.abs(x), min=1e-6)
                            # sum_across_bsz = abs_x.view(cfg['probe_num'], cfg['probe_size'], x.size(-2), x.size(-1)).sum(dim=1, keepdim=True)
                            # proportion = abs_x.view(cfg['probe_num'], cfg['probe_size'], x.size(-2), x.size(-1)) / sum_across_bsz
                            # comp_across_bsz = (x.view(cfg['probe_num'], cfg['probe_size'], x.size(-2), x.size(-1)) * proportion).sum(dim=1)

                            end_time = time.time()
                            has_nan = torch.isnan(comp_across_bsz).any()
                            if has_nan:
                                print(f"Does 'comp_across_bsz' contain NaN values? {has_nan}")
                            # print(f"Does 'comp_across_bsz' contain NaN values? {has_nan}")
                            print('nml_duration2', end_time - start_time, comp_across_bsz.shape, flush=True)

                            
                        
                        # print('isequal', comp_across_bsz == x, comp_across_bsz, x)
                        # comp_across_bsz = x
                        gate_out = None
                        up_out = None
                        if 'gate_proj' in cfg['cust_tgt_modules']:
                            if 'svd' in cfg['prune_method']:
                                print('svd')
                                gate_out = self.act_fn(x @ self.gate_proj_svd_V.T @ self.gate_proj_svd_S.T @ self.gate_proj_svd_U.T)
                            else:
                                gate_out = self.act_fn(self.gate_proj(comp_across_bsz, cal_mlp_probe_out_dim_metric=True))
                        else:
                            gate_out = self.act_fn(self.gate_proj(x))
                        
                        if 'up_proj' in cfg['cust_tgt_modules']:
                            if 'svd' in cfg['prune_method']:
                                print('svd')
                                up_out = x @ self.up_proj_svd_V.T @ self.up_proj_svd_S.T @ self.up_proj_svd_U.T
                            else:
                                # print('cal_mlp_probe_out_dim_metric')
                                up_out = self.up_proj(comp_across_bsz, cal_mlp_probe_out_dim_metric=True)
                        else:
                            up_out = self.up_proj(x)

                        probe_out = gate_out * up_out
                        if 'twoprobe' in cfg['prune_method']:
                            probe_one = probe_out[:x.shape[1], :].unsqueeze(0)
                            print('probe_one twoprobe', probe_one.shape, flush=True)
                            probe_two = probe_out[x.shape[1]:, :].unsqueeze(1)
                            print('probe_two twoprobe', probe_two.shape, flush=True)
                            probe_out = probe_one * probe_two
                            print('probe_out twoprobe', probe_out.shape, flush=True)

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
                        if self.intersected_prune_indices is not None:
                            probe_out[..., self.intersected_prune_indices] = 0

                        if 'async' in cfg['prune_method'] and 'savemetricseq' in cfg['prune_method']:
                            # temp = probe_out
                            # probe_out = self.last_batch_probe_out
                            # self.last_batch_probe_out = temp

                            if 'squareasync' in cfg['prune_method']:
                                if self.last_batch_probe_out is None:
                                    norm_probe_out_square = torch.clamp(torch.norm(probe_out, p=2, dim=0) ** 2, min=None, max=65504) / cfg['probe_num']
                                    self.last_batch_probe_out = norm_probe_out_square.detach()
                                    probe_out = torch.zeros(1, probe_out.size(-2), probe_out.size(-1), device=probe_out.device, dtype=probe_out.dtype)
                                else:
                                # self.last_batch_probe_out = self.last_batch_probe_out.to(probe_out.device)    
                                    # if 'squareasyncabs' in cfg['prune_method']:
                                    #     probe_out = torch.norm(probe_out, p=2, dim=0) ** 2
                                    #     abs_probe = torch.abs(probe_out).to(torch.float32)
                                    #     abs_last_batch_probe = torch.abs(self.last_batch_probe_out).to(torch.float32)
                                    #     sum_across_two_terms = abs_probe + abs_last_batch_probe
                                    #     # proportion = abs_x / torch.sum(abs_x, dim=0, keepdim=True)
                                    #     proportion = (abs_probe / (sum_across_two_terms + 1e-10)).to(abs_probe.dtype)
                                    #     combined_probe = (probe_out * proportion + self.last_batch_probe_out * (1 - proportion)).to(probe_out.dtype)
                                    #     probe_out, self.last_batch_probe_out = torch.sqrt(self.last_batch_probe_out).unsqueeze(0), combined_probe
                                    # if 'squareasync' in cfg['prune_method']:
                                    proportion = cfg['asyncratio']
                                    probe_out = torch.clamp(torch.norm(probe_out, p=2, dim=0) ** 2, min=None, max=65504) / cfg['probe_num']
                                    combined_probe = (self.last_batch_probe_out * proportion + probe_out * (1 - proportion)).to(probe_out.dtype)
                                    # combined_probe = probe_out
                                    probe_out, self.last_batch_probe_out = torch.sqrt(self.last_batch_probe_out).unsqueeze(0), combined_probe
                                    # proportion = 10
                                    # print('proportion ', proportion, flush=True)
                                    # comp_across_bsz = torch.sum(x * proportion, dim=0)
                                    # comp_across_bsz = comp_across_bsz.unsqueeze(0)
                                    
                                    # combined_probe = (probe_out * 0.5 + self.last_batch_probe_out * (1 - 0.5)).to(probe_out.dtype)
                                    print('combined_probe', combined_probe.shape, flush=True)
                                    
                                    print('probe_out shape', probe_out.shape, flush=True)
                            else:
                                if self.last_batch_probe_out is None:
                                    self.last_batch_probe_out = probe_out
                                    probe_out = torch.zeros_like(probe_out)
                                else:
                                # self.last_batch_probe_out = self.last_batch_probe_out.to(probe_out.device)
                                    # if 'asyncabs' in cfg['prune_method']:
                                    #     abs_probe = torch.abs(probe_out).to(torch.float32)
                                    #     abs_last_batch_probe = torch.abs(self.last_batch_probe_out).to(torch.float32)
                                    #     sum_across_two_terms = abs_probe + abs_last_batch_probe
                                    #     # proportion = abs_x / torch.sum(abs_x, dim=0, keepdim=True)
                                    #     proportion = (abs_probe / (sum_across_two_terms + 1e-10)).to(abs_probe.dtype)
                                    #     combined_probe = (probe_out * proportion + self.last_batch_probe_out * (1 - proportion)).to(probe_out.dtype)
                                    #     probe_out, self.last_batch_probe_out = self.last_batch_probe_out, combined_probe
                                    # else:
                                    proportion = cfg['asyncratio']
                                    combined_probe = (self.last_batch_probe_out * proportion + probe_out * (1 - proportion)).to(probe_out.dtype)
                                    probe_out, self.last_batch_probe_out = self.last_batch_probe_out, combined_probe
                                        # proportion = 10
                                    # print('proportion ', proportion, flush=True)
                                    # comp_across_bsz = torch.sum(x * proportion, dim=0)
                                    # comp_across_bsz = comp_across_bsz.unsqueeze(0)
                                    # combined_probe = (probe_out * proportion + self.last_batch_probe_out * (1 - proportion)).to(probe_out.dtype)
                                    # probe_out, self.last_batch_probe_out = self.last_batch_probe_out, combined_probe

                        if 'calib' in cfg['prune_method'] or 'runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']:
                            # if 'saveseqdim' in cfg['prune_method']:
                            #     probe_out_dim_metric, comined_probe_out = cal_prune_metric(probe_out, self.down_proj.weight.data, cfg['prune_metric'], global_input_distribution=self.down_proj.get_global_input_distribution()[0])
                            # else:
                            probe_out_dim_metric, comined_probe_out = cal_prune_metric(probe_out, self.down_proj.weight.data, cfg['prune_metric'], global_metric_score_distribution=self.down_proj.get_global_metric_score_distribution())
                        else:
                            probe_out_dim_metric, comined_probe_out = cal_prune_metric(probe_out, self.down_proj.weight.data, cfg['prune_metric'])

                        if 'globalratio' in cfg['prune_method']:
                            probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_probe_mlp_metric(probe_out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.down_proj.pruning_ratio)
                        else:
                            probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_probe_mlp_metric(probe_out_dim_metric, cfg['tc_multiple'])

                        # if 'dynaprobe' in cfg['prune_method']:
                        #     global_score = copy.deepcopy(self.down_proj.get_global_metric_score_distribution())
                        #     if global_score.dim() == 1:
                        #         global_score = global_score.unsqueeze(0).unsqueeze(0)
                        #     elif global_score.dim() == 2:
                        #         global_score = global_score.unsqueeze(0)

                        #     raw_calib_out_dim_metric, _ = cal_prune_metric(global_score, self.down_proj.weight.data, cfg['prune_metric'])
                        #     raw_calib_out_dim_indices, raw_calib_prune_dim_indices = self.pruning_module.sort_probe_mlp_metric(raw_calib_out_dim_metric, multiple)

                        #     raw_calib_set = set(raw_calib_out_dim_indices.tolist())
                        #     probe_out_set = set(probe_out_dim_indices.tolist())

                        #     intersection_ratio = len(raw_calib_set & probe_out_set) / len(probe_out_set)
                        #     print('calib_and_plus_calib_intersection_ratio', intersection_ratio, flush=True)

                        #     # Find elements in probe_out_dim_indices that are missing in raw_calib_out_dim_indices
                        #     missing_elements = probe_out_set - raw_calib_set

                        #     # To find the original positions of the missing elements in probe_out_dim_indices
                        #     missing_positions = [i for i, element in enumerate(probe_out_dim_indices.tolist()) if element in missing_elements]

                        #     print('Positions of elements in probe_out_dim_indices missing in raw_calib_out_dim_indices:', missing_positions)

                        #     raw_prune_calib_set = set(raw_calib_prune_dim_indices.tolist())
                        #     probe_prune_set = set(prune_out_dim_indices.tolist())

                        #     intersected_prune_indices_list = list(raw_prune_calib_set & probe_prune_set)

                        #     # Convert the list to a PyTorch tensor
                        #     self.intersected_prune_indices = torch.tensor(intersected_prune_indices_list, dtype=torch.long, device=probe_out.device)
                        #     intersection_ratio = len(raw_prune_calib_set & probe_prune_set) / len(probe_prune_set)
                        #     print('pruned calib_and_plus_calib_intersection_ratio', intersection_ratio, flush=True)


                        #     sorted_value, sorted_indices = torch.sort(probe_out_dim_metric, dim=0)
                        #     # torch.set_printoptions(threshold=5000)
                            # torch.set_printoptions(threshold=5000, edgeitems=2000)
                            # print('sorted_value', sorted_value, flush=True)
                        # if self.probe_out_dim_indices is None:
                        #     self.probe_out_dim_indices = probe_out_dim_indices
                        # else:
                        #     # Convert lists to sets
                        #     set_self = set(self.probe_out_dim_indices.tolist())
                        #     set_probe = set(probe_out_dim_indices.tolist())

                        #     # Find the intersection
                        #     intersection = set_self & set_probe

                        #     # Count the number of elements in the intersection
                        #     intersection_count = len(intersection)
                        #     intersection_ratio = intersection_count / len(set_self)
                        #     print('intersection_ratio', intersection_ratio, flush=True)
                        #     self.probe_out_dim_indices = probe_out_dim_indices

                        
                        custom_duration = time.time() - time_start
                        # print('probe_duration', custom_duration, flush=True)
                        time_start = time.time()
                        if 'gate_proj' in cfg['cust_tgt_modules']:
                            gate_out = self.act_fn(self.gate_proj(x, probe_out_dim_indices=probe_out_dim_indices))
                        else:
                            # gate_out = self.act_fn(self.gate_proj(x))
                            gate_out = gate_out[..., probe_out_dim_indices]

                        if 'up_proj' in cfg['cust_tgt_modules']:
                            up_out = self.up_proj(x, probe_out_dim_indices=probe_out_dim_indices)
                        else:
                            # up_out = self.up_proj(x)
                            up_out = up_out[..., probe_out_dim_indices]

                        # intermediate_output = 
                        
                            # self.down_proj.update_global_metric_score_distribution(intermediate_output[..., probe_out_dim_indices], probe_out_dim_indices)
                            # fill the probe predict for prune_out_dim_indices
                        if 'fillpbmetric' in cfg['prune_method']:
                                # self.down_proj.update_global_metric_score_distribution(probe_out[..., prune_out_dim_indices], prune_out_dim_indices, batch_size=bsz, is_probe=True)
                                # Selecting specific dimensions
                            # .expand(bsz, -1, -1)
                            if 'runningmean' in cfg['prune_method']:
                                self.down_proj.update_global_metric_score_distribution(probe_out[..., prune_out_dim_indices], prune_out_dim_indices)
                            elif 'ema' in cfg['prune_method']:
                                if 'fillpbmetricoriginal' in cfg['prune_method']:
                                    self.down_proj.update_global_metric_score_distribution_ema(probe_out[..., prune_out_dim_indices], prune_out_dim_indices, is_probe=True)

                                    # bsz_tensor = torch.tensor(bsz, dtype=torch.float16, device=gate_out.device)
                                    # # real_out_channel = torch.clamp(torch.norm(gate_out * up_out, p=2, dim=0) ** 2, min=None, max=65504) / torch.sqrt(bsz_tensor)
                                    # probe_select = torch.clamp(torch.norm(probe_out[..., probe_out_dim_indices], p=2, dim=0) ** 2, min=None, max=65504)
                                    # prune_prune = torch.clamp(torch.norm(probe_out[..., prune_out_dim_indices], p=2, dim=0) ** 2, min=None, max=65504)
                                    # full_inference = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                                    # full_inference = torch.clamp(torch.norm(full_inference, p=2, dim=0) ** 2, min=None, max=65504) / bsz_tensor
                                    # full_selected = full_inference[..., probe_out_dim_indices]
                                    # full_pruned = full_inference[..., prune_out_dim_indices]

                                    # print('\nself.layer_order', self.layer_order)
                                    # gap_between_real_probe = full_selected - probe_select
                                    # print('gap_between_real_probe', gap_between_real_probe, flush=True)
                                    # gap_between_real_probe_mean = gap_between_real_probe.mean()
                                    # print('gap_between_real_probe_mean', gap_between_real_probe_mean, flush=True)
                                    # gap_between_real_probe_std = gap_between_real_probe.std()
                                    # print('gap_between_real_probe_std', gap_between_real_probe_std, flush=True)

                                    # gap_norm_over_seq = torch.norm(gap_between_real_probe, p=2, dim=0)
                                    # sorted_gap_norm_over_seq, sorted_indices = torch.sort(gap_norm_over_seq)

                                    # gap_norm_over_dim = torch.norm(gap_between_real_probe, p=2, dim=1)
                                    # sorted_gap_norm_over_dim, sorted_indices_over_dim = torch.sort(gap_norm_over_dim)
                                    # print('sorted_gap_norm_over_seq', sorted_gap_norm_over_seq, flush=True)
                                    # print('sorted_gap_norm_over_dim', sorted_gap_norm_over_dim, flush=True)
                                    # print('sorted_indices_over_dim', sorted_indices_over_dim, flush=True)

                                    # print('prune---------')
                                    # gap_between_real_probe = full_pruned - prune_prune
                                    # print('gap_between_real_probe', gap_between_real_probe, flush=True)
                                    # gap_between_real_probe_mean = gap_between_real_probe.mean()
                                    # print('gap_between_real_probe_mean', gap_between_real_probe_mean, flush=True)
                                    # gap_between_real_probe_std = gap_between_real_probe.std()

                                    # gap_norm_over_seq = torch.norm(gap_between_real_probe, p=2, dim=0)
                                    # sorted_gap_norm_over_seq, sorted_indices = torch.sort(gap_norm_over_seq)

                                    # gap_norm_over_dim = torch.norm(gap_between_real_probe, p=2, dim=1)
                                    # sorted_gap_norm_over_dim_prune, sorted_indices_over_dim_prune = torch.sort(gap_norm_over_dim)
                                    # print('sorted_gap_norm_over_dim_prune', sorted_gap_norm_over_dim_prune, flush=True)
                                    # print('sorted_indices_over_dim_prune', sorted_indices_over_dim_prune, flush=True)

                                    # def compare_accumulated_deciles_set_match(indices1, indices2):
                                    #     assert len(indices1) == len(indices2), "Indices must be of the same length"
                                        
                                    #     decile_size = len(indices1) // 10
                                    #     match_ratios = []

                                    #     for i in range(10):
                                    #         end_idx = (i + 1) * decile_size if i < 9 else len(indices1)
                                            
                                    #         # Extract the accumulated segments as Python sets
                                    #         decile1_set = set(indices1[:end_idx].tolist())
                                    #         decile2_set = set(indices2[:end_idx].tolist())
                                            
                                    #         # Count set matches
                                    #         matches = len(decile1_set.intersection(decile2_set))
                                            
                                    #         # Calculate the match ratio based on the number of unique elements in decile1
                                    #         match_ratio = matches / len(decile1_set)
                                    #         match_ratios.append(match_ratio)
                                        
                                    #     return match_ratios

                                    # match_ratios = compare_accumulated_deciles_set_match(sorted_indices_over_dim, sorted_indices_over_dim_prune)

                                    # for i, ratio in enumerate(match_ratios, start=1):
                                    #     print(f"Accumulated Decile {i*10}%: {ratio*100:.2f}% match")

                                    # full_inference_pruned = full_inference[..., prune_out_dim_indices]
                                    # gap_between_real_probe_norm = gap_between_real_probe.max()
                                elif 'fillpbmetriccombine' in cfg['prune_method']:
                                    self.down_proj.update_global_metric_score_distribution_ema(comined_probe_out[..., prune_out_dim_indices], prune_out_dim_indices, is_probe=True)
                                elif 'fillpbmetricub' in cfg['prune_method']:
                                    full_inference = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                                    # full_inference = torch.clamp(torch.norm(full_inference, p=2, dim=0) ** 2, min=None, max=65504) / bsz_tensor
                                    # full_selected = full_inference[..., probe_out_dim_indices]
                                    full_pruned = full_inference[..., prune_out_dim_indices]
                                    self.down_proj.update_global_metric_score_distribution_ema(full_pruned, prune_out_dim_indices)

                        if 'halfsquareasync' in cfg['prune_method'] and 'savemetricseq' in cfg['prune_method']:
                            temp_norm_square = torch.clamp(torch.norm(gate_out * up_out, p=2, dim=0) ** 2, min=None, max=65504) / bsz
                            self.last_batch_probe_out[..., probe_out_dim_indices] = temp_norm_square

                        kwargs['probe_in_dim_indices'] = probe_out_dim_indices
                        down_proj = self.down_proj(gate_out * up_out, **kwargs)
                        custom_duration = time.time() - time_start
                        # print('fll_batch_duration', custom_duration, flush=True)
                        return down_proj
                    elif 'calib' in cfg['prune_method'] and ('runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']):
                        bsz, _, _ = x.shape
                        time_start = time.time()

                        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

                        with torch.cuda.stream(cfg['cuda_stream1']):
                            if torch.all(self.down_proj.get_global_metric_score_distribution() == 0):
                                out_dim_indices = torch.arange(self.intermediate_size, dtype=torch.long).to(device=x.device)
                            else:
                                out_dim_metric = cal_calib_prune_metric(self.down_proj.get_global_metric_score_distribution(), self.down_proj.weight.data, cfg['prune_metric'])

                                if 'globalratio' in cfg['prune_method']:
                                    out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_probe_mlp_metric(out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.down_proj.pruning_ratio)
                                else:
                                    out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_probe_mlp_metric(out_dim_metric, cfg['tc_multiple'])

                            self.gate_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                            self.up_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                            self.down_proj.prepare_async_weight(in_dim_indices=out_dim_indices)
                        return down_proj
                    elif 'calib' in cfg['prune_method']:
                        bsz, _, _ = x.shape
                        time_start = time.time()

                        if cfg['mode'] == 'sync':
                            if torch.all(self.down_proj.get_global_metric_score_distribution() == 0):
                                out_dim_indices = torch.arange(self.intermediate_size, dtype=torch.long).to(device=x.device)
                            else:
                                out_dim_metric = cal_calib_prune_metric(self.down_proj.get_global_metric_score_distribution(), self.down_proj.weight.data, cfg['prune_metric'])

                                if 'globalratio' in cfg['prune_method']:
                                    out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_probe_mlp_metric(out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.down_proj.pruning_ratio)
                                else:
                                    out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_probe_mlp_metric(out_dim_metric, cfg['tc_multiple'])


                            down_proj = self.down_proj(self.act_fn(self.gate_proj(x, out_dim_indices=out_dim_indices)) * self.up_proj(x, out_dim_indices=out_dim_indices), in_dim_indices=out_dim_indices)

                        if cfg['mode'] == 'asyncinter':
                            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

                            if cfg['cur_batch_index'] == 0:
                                with torch.cuda.stream(cfg['cuda_stream1']):
                                    if torch.all(self.down_proj.get_global_metric_score_distribution() == 0):
                                        out_dim_indices = torch.arange(self.intermediate_size, dtype=torch.long).to(device=x.device)
                                    else:
                                        out_dim_metric = cal_calib_prune_metric(self.down_proj.get_global_metric_score_distribution(), self.down_proj.weight.data, cfg['prune_metric'])

                                        if 'globalratio' in cfg['prune_method']:
                                            out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_probe_mlp_metric(out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.down_proj.pruning_ratio)
                                        else:
                                            out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_probe_mlp_metric(out_dim_metric, cfg['tc_multiple'])

                                    self.gate_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                                    self.up_proj.prepare_async_weight(out_dim_indices=out_dim_indices)
                                    self.down_proj.prepare_async_weight(in_dim_indices=out_dim_indices)
                        
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
                    #         probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_probe_mlp_metric(probe_out_dim_metric, multiple)

                    #     temp = self.act_fn(self.gate_proj(x, probe_out_dim_indices=probe_out_dim_indices)) * self.up_proj(x, probe_out_dim_indices=probe_out_dim_indices)
                    #     # if 'runningmean' in cfg['prune_method']:
                    #         # print('runningmean', flush=True)
                    #         # print('probe_out_dim_indices', probe_out_dim_indices, flush=True)
                    #         # print('temp', temp, flush=True)
                    #         # self.down_proj.update_global_metric_score_distribution(temp, probe_out_dim_indices)
                    #         # Update the running_mean and running_mean_counter here
                    #         # self.running_mean[probe_out_dim_indices] *= self.running_mean_counter[probe_out_dim_indices] / (self.running_mean_counter[probe_out_dim_indices] + bsz)
                    #         # # Ensure the denominator is broadcastable; might need to unsqueeze to add a dimension for correct broadcasting
                    #         # norm_squared = torch.clamp(torch.norm(temp, p=2, dim=1) ** 2, min=None, max=65504)
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
        self.cal_total_flops = True
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

        self.q_proj.cal_total_flops = True
        self.k_proj.cal_total_flops = True
        self.v_proj.cal_total_flops = True
        self.o_proj.cal_total_flops = True

        self.layer_order = layer_order
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
                # if 'calib' in cfg['prune_method'] and ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']):
                time_start = time.time()
                # full inference
                bsz, q_len, _ = hidden_states.size()

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

                # query: bsz, self.num_heads, 1, self.head_dim
                # key_states: bsz, self.num_key_value_heads, q_len+1, self.head_dim -> bsz, self.num_key_value_heads, self.head_dim, q_len+1
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
                attn_output = self.o_proj(attn_output)

                if not output_attentions:
                    attn_weights = None

                custom_duration = time.time() - time_start
                return attn_output, attn_weights, past_key_value
            elif cfg['calibration_stage'] == False :
                bsz, q_len, _ = hidden_states.size()
                if 'probe' in cfg['prune_method']:
                    if 'nml' in cfg['prune_method']:

                        # comp_across_bsz_q = nml_process(hidden_states, cfg['q_probe_num'], cfg['q_probe_size'])
                        # comp_across_bsz_k = nml_process(hidden_states, cfg['k_probe_num'], cfg['k_probe_size'])
                        # comp_across_bsz_v = nml_process(hidden_states, cfg['v_probe_num'], cfg['v_probe_size'])
                        comp_across_bsz_q = nml_process(kwargs['temp_mlp_residual'], cfg['q_probe_num'], cfg['q_probe_size'])
                        comp_across_bsz_k = nml_process(kwargs['temp_mlp_residual'], cfg['k_probe_num'], cfg['k_probe_size'])
                        comp_across_bsz_v = nml_process(kwargs['temp_mlp_residual'], cfg['v_probe_num'], cfg['v_probe_size'])
                    else:
                        comp_across_bsz_q = hidden_states.mean(axis=0).unsqueeze_(0)
                        comp_across_bsz_k = hidden_states.mean(axis=0).unsqueeze_(0)
                        comp_across_bsz_v = hidden_states.mean(axis=0).unsqueeze_(0)

                    print('comp_across_bsz_v', comp_across_bsz_v.shape, flush=True)
                    q_num_heads, k_num_heads, v_num_heads = self.num_heads, self.num_key_value_heads, self.num_key_value_heads
                    q_head_dim, k_head_dim, v_head_dim = self.head_dim, self.head_dim, self.head_dim

                    # copy orignal code and modify a little bit for probe pruning
                    # currently does not implement for group attention, but it should work too
                    # bsz, q_len, _ = comp_across_bsz_qk.size()
                    query_states = self.q_proj(comp_across_bsz_q, cal_attn_probe_out_dim_metric=True)   
                    key_states = self.k_proj(comp_across_bsz_k, cal_attn_probe_out_dim_metric=True)
                    value_states = self.v_proj(comp_across_bsz_v, cal_attn_probe_out_dim_metric=True)
                    
                    # key_states = repeat_kv(key_states, self.num_key_value_groups)
                    # value_states = repeat_kv(value_states, self.num_key_value_groups)

                    # only consider qk effect, v and o remain the same
                    # if 'onlyqk' in cfg['prune_name']:
                    # if 'compressseq' in cfg['prune_metric']:
                    #     if 'mbsz' in cfg['prune_metric']:
                    #         temp_comp_across_bsz_qk = hidden_states.mean(axis=0)
                    #         temp_comp_across_bsz_qk = torch.linalg.vector_norm(temp_comp_across_bsz_qk, ord=2, dim=0).unsqueeze_(0)
                    #     elif 'mbszmseq' in cfg['prune_metric']:
                    #         temp_comp_across_bsz_qk = hidden_states.mean(axis=(0,1)).unsqueeze_(0)
                    #     else:
                    #         temp_comp_across_bsz_qk = torch.linalg.vector_norm(hidden_states, ord=2, dim=(0,1)).unsqueeze_(0)

                    #     temp_query_states = self.q_proj(temp_comp_across_bsz_qk, cal_attn_probe_out_dim_metric_compressseq='q_proj' in cfg['cust_tgt_modules'])
                    #     temp_key_states = self.k_proj(temp_comp_across_bsz_qk, cal_attn_probe_out_dim_metric_compressseq='k_proj' in cfg['cust_tgt_modules'])

                    #     if 'wandasp' in cfg['prune_metric']:
                    #         probe_qk_out_dim_metric = (torch.linalg.vector_norm(temp_query_states, ord=2, dim=(0, 1)).reshape((1, 1, -1)) * torch.abs(temp_query_states)).sum(axis=(0, 1))
                    #     elif 'flap' in cfg['prune_metric']:
                    #         pass
                    #     elif 'probe' in cfg['prune_metric']:
                    #         temp_query_states = temp_query_states.to(torch.float32)
                    #         temp_key_states = temp_key_states.to(torch.float32)
                    #         # probe_out_dim_metric = torch.sqrt((torch.sum(torch.pow(probe_out, 2), dim=1).reshape((1, -1)) * torch.pow(self.down_proj.weight.data, 2)).sum(axis=0))
                    #         probe_qk_out_dim_metric = torch.sqrt((torch.sum(torch.pow(temp_query_states, 2), dim=1).reshape((-1, 1)) * torch.pow(temp_key_states, 2)).sum(axis=1))
                    #         # query_states = query_states.to(hidden_states.dtype)
                    #         # key_states = key_states.to(hidden_states.dtype)

                    # else:
                    qk_prune_way = cfg['qk_prune_way']
                    vo_prune_way = cfg['vo_prune_way']

                    # TODO: sep prune
                    # if qk_prune_way is not None:
                    #     if 'wandasp' in cfg['prune_metric']:
                    #         probe_qk_out_dim_metric = (torch.linalg.vector_norm(query_states, ord=2, dim=(0, 1)).reshape((1, 1, -1)) * torch.abs(key_states)).sum(axis=(0, 1))
                    #     elif 'flap' in cfg['prune_metric']:
                    #         pass
                    #     elif 'probe' in cfg['prune_metric']:
                    #         query_states = query_states.to(torch.float32)
                    #         key_states = key_states.to(torch.float32)
                    #         probe_qk_out_dim_metric = torch.sqrt((torch.sum(torch.pow(query_states, 2), dim=(0, 1)).reshape((1, 1, -1)) * torch.pow(key_states, 2)).sum(axis=(0, 1)))
                    #         query_states = query_states.to(hidden_states.dtype)
                    #         key_states = key_states.to(hidden_states.dtype)

                    #     probe_qk_out_dim_indices, probe_qk_out_dim_indices_for_rope, qk_num_heads, qk_head_dim = self.pruning_module.sort_probe_attn_metric(probe_qk_out_dim_metric, self.num_heads, self.head_dim, qk_prune_way, 'qk', multiple)
                    #     q_num_heads, k_num_heads = qk_num_heads, qk_num_heads
                    #     q_head_dim, k_head_dim = qk_head_dim, qk_head_dim

                    # if 'conditionqk' in cfg['prune_name']:
                    #     # query_states = self.q_proj(hidden_states, probe_out_dim_indices=probe_qk_out_dim_indices)
                    #     # key_states = self.k_proj(hidden_states, probe_out_dim_indices=probe_qk_out_dim_indices)
                    #     if 'fill' in cfg['qk_proj_prune']:
                    #         mask = torch.zeros(query_states.shape[-1], dtype=torch.bool, device=query_states.device)
                    #         mask[~probe_qk_out_dim_indices] = True
                    #         query_states[..., mask] = 0
                    #         key_states[..., mask] = 0
                    #     else:
                    #         query_states = torch.index_select(query_states, -1, probe_qk_out_dim_indices)
                    #         key_states = torch.index_select(key_states, -1, probe_qk_out_dim_indices)
                            
                    #     query_states = query_states.view(bsz, q_len, q_num_heads, q_head_dim).transpose(1, 2)
                    #     key_states = key_states.view(bsz, q_len, k_num_heads, k_head_dim).transpose(1, 2)
                    # else:
                    query_states = query_states.view(query_states.shape[0], q_len, self.num_heads, self.head_dim).transpose(1, 2)
                    key_states = key_states.view(key_states.shape[0], q_len, self.num_heads, self.head_dim).transpose(1, 2)
                    value_states = value_states.view(value_states.shape[0], q_len, self.num_heads, self.head_dim).transpose(1, 2)

                    kv_seq_len = key_states.shape[-2]
                    # if past_key_value is not None:
                    #     kv_seq_len += past_key_value[0].shape[-2]
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

                    # if 'each' in qk_prune_way:
                    #     query_states, key_states = apply_rotary_pos_emb_for_prune_each_head(query_states, key_states, cos, sin, position_ids, probe_qk_out_dim_indices_for_rope)
                    # else:
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

                    # key_states = repeat_kv(key_states, self.num_key_value_groups)
                    # value_states = repeat_kv(value_states, self.num_key_value_groups)

                    # key_states: bsz, self.num_key_value_heads, q_len, self.head_dim -> bsz, self.num_key_value_heads, self.head_dim, q_len
                    # if 'conditionqk' in cfg['prune_method']:
                    #     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(k_head_dim)
                    # else:
                    
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

                    # .expand(cfg['k_probe_num'], -1, -1, -1)
                    print('attention_mask', attention_mask.shape, flush=True)
                    probe_attn_mask = attention_mask[:cfg['k_probe_num'], ...]
                    print('probe_attn_mask', probe_attn_mask.shape, flush=True)
                    # print('probe_attn_mask', probe_attn_mask, flush=True)
                    
                    # if 'beforedeleteseq' in cfg['prune_name']:
                    #     zero_attn_mask = (probe_attn_mask >= 0)
                    #     # print('zero_attn_mask', zero_attn_mask, flush=True)
                    #     attn_weights = attn_weights * zero_attn_mask
                    #     # print('attn_weightsaftermask', attn_weights, attn_weights.shape, flush=True)
                    #     if cfg['prune_metric'] == 'WOF2N' or cfg['prune_metric'] == 'mbWOF2N':
                    #         # becomes self.num_heads, q_len
                    #         attn_weights_metric = (torch.linalg.vector_norm(attn_weights, ord=2, dim=(0, 2)).reshape((1, self.num_heads, -1, 1)) * torch.abs(value_states)).sum(axis=(0, -1))
                    #     elif cfg['prune_metric'] == 'WOF2S' or cfg['prune_metric'] == 'mbWOF2S':
                    #         # becomes self.num_heads, q_len
                    #         attn_weights_metric = torch.sqrt((torch.sum(torch.pow(attn_weights, 2), dim=(0, 2)).reshape((1, self.num_heads, -1, 1)) * torch.pow(value_states, 2)).sum(axis=(0, -1)))

                        # attn_weights_indices = self.pruning_module.cal_probe_attn_weights_metric(attn_weights_metric)
                    if probe_attn_mask is not None:
                        if probe_attn_mask.size() != (cfg['k_probe_num'], 1, q_len, kv_seq_len):
                            raise ValueError(
                                f"Attention mask should be of size {(cfg['k_probe_num'], 1, q_len, kv_seq_len)}, but is {probe_attn_mask.size()}"
                            )
                        attn_weights = attn_weights + probe_attn_mask
                        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
                        # print('attnweightsaftermask', attn_weights, attn_weights.shape, flush=True)

                    # upcast attention to fp32
                    # attn_weights: bsz, self.num_heads, q_len, q_len
                    # attn_weights: 1, self.num_heads, q_len, q_len
                    # value_states: 1, self.num_key_value_heads, q_len, self.head_dim
                    
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    # TODO: fix later
                    if 'delseq' in cfg['prune_method']:
                        if 'wandasp' in cfg['prune_metric']:
                            # becomes self.num_heads, q_len
                            attn_weights_metric = (torch.linalg.vector_norm(attn_weights, ord=2, dim=(0, 2)).reshape((1, self.num_heads, -1, 1)) * torch.abs(value_states)).sum(axis=(0, -1))
                        elif 'flap' in cfg['prune_metric']:
                            pass
                        elif 'probe' in cfg['prune_metric']:
                            # becomes self.num_heads, q_len
                            # reshape to do element-wise multiplication
                            attn_weights_metric = torch.sqrt((torch.sum(torch.pow(attn_weights, 2), dim=(0, 2)).reshape((1, self.num_heads, -1, 1)) * torch.pow(value_states, 2)).sum(axis=(0, -1)))

                        # dont customize the metric for each head
                        # if cfg['prune_metric'] == 'WOF2N' or cfg['prune_metric'] == 'mbWOF2N':
                        #     # becomes self.num_heads, q_len
                        #     attn_weights_metric = (torch.linalg.vector_norm(attn_weights, ord=2, dim=(0, 1, 2)).reshape((1, 1, -1, 1)) * torch.abs(value_states)).sum(axis=(0, 1, 3))
                        # elif cfg['prune_metric'] == 'WOF2S' or cfg['prune_metric'] == 'mbWOF2S':
                        #     # becomes self.num_heads, q_len
                        #     attn_weights_metric = torch.sqrt((torch.sum(torch.pow(attn_weights, 2), dim=(0, 1, 2)).reshape((1, 1, -1, 1)) * torch.pow(value_states, 2)).sum(axis=(0, 1, 3)))
                        _, attn_weights_indices, _, _ = self.pruning_module.sort_probe_attn_metric(attn_weights_metric, attn_weights.shape[1], 1, cfg['vo_proj_prune'], 'delseq', cfg['tc_multiple'])

                    # print('attn_weightsafter softmax', attn_weights, attn_weights.shape, flush=True)
                    # value_states: bsz, self.num_key_value_heads, q_len, self.head_dim
                    # value_states: 1, self.num_key_value_heads, q_len, self.head_dim
                    print('attn_weights_shape', attn_weights.shape, flush=True)
                    print('value_states_shape', value_states.shape, flush=True)
                    if cfg['k_probe_num'] > cfg['v_probe_num']:
                        value_states = value_states.repeat_interleave(cfg['k_probe_num']// cfg['v_probe_num'], dim=0)
                    else:
                        attn_weights = attn_weights.repeat_interleave(cfg['v_probe_num']// cfg['k_probe_num'], dim=0)
                    print('value_states_shape', value_states.shape, flush=True)
                    attn_output = torch.matmul(attn_weights, value_states)
                    print('attn_output_shape', attn_output.shape, flush=True)
                    
                    # print('attn_output00', attn_output[0][0], attn_output[0][0].shape, flush=True)
                    if attn_output.size() != (max(cfg['k_probe_num'], cfg['v_probe_num']), v_num_heads, q_len, v_head_dim):
                        raise ValueError(
                            f"`attn_output` should be of size {(max(cfg['k_probe_num'], cfg['v_probe_num']), v_num_heads, q_len, v_head_dim)}, but is"
                            f" {attn_output.size()}"
                        )

                    # attn_output: bsz, self.num_heads, q_len, self.head_dim -> bsz, q_len, self.num_heads, self.head_dim
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(max(cfg['k_probe_num'], cfg['v_probe_num']), q_len, self.hidden_size)
                    
                    if 'calib' in cfg['prune_method'] or 'runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']:
                            # if 'saveseqdim' in cfg['prune_method']:
                            #     probe_out_dim_metric, comined_probe_out = cal_prune_metric(probe_out, self.down_proj.weight.data, cfg['prune_metric'], global_input_distribution=self.down_proj.get_global_input_distribution()[0])
                            # else:
                        probe_out_dim_metric, comined_probe_out = cal_prune_metric(attn_output, self.o_proj.weight.data, cfg['prune_metric'], global_metric_score_distribution=self.o_proj.get_global_metric_score_distribution())
                    else:
                        probe_out_dim_metric, comined_probe_out = cal_prune_metric(attn_output, self.o_proj.weight.data, cfg['prune_metric'])

                    if 'globalratio' in cfg['prune_method']:
                        probe_vo_out_dim_indices, probe_vo_out_dim_indices_for_rope, v_num_heads, v_head_dim = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, v_num_heads, v_head_dim, vo_prune_way, 'vo', cfg['tc_multiple'], pruning_ratio=self.o_proj.pruning_ratio)
                    else:
                        # probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_probe_mlp_metric(probe_out_dim_metric, multiple)
                        probe_vo_out_dim_indices, probe_vo_out_dim_indices_for_rope, v_num_heads, v_head_dim = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, v_num_heads, v_head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])
                    
                    if 'whole' in vo_prune_way:
                        probe_qk_out_dim_indices = probe_vo_out_dim_indices
                        qk_prune_way = vo_prune_way
                        q_num_heads, k_num_heads = v_num_heads, v_num_heads
                        q_head_dim, k_head_dim = v_head_dim, v_head_dim
                        # probe_out_dim_indices, _, v_num_heads, v_head_dim = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, v_num_heads, v_head_dim, vo_prune_way, 'vo', multiple)

                    # if 'restore' in cfg['prune_name']:
                    #     temp = torch.arange(4096).to(dtype=probe_out_dim_indices.dtype, device=probe_out_dim_indices.device)
                    #     prune_out_dim_indices = temp[~probe_out_dim_indices]
                    #     restore = F.linear(attn_output[..., prune_out_dim_indices], torch.index_select(self.o_proj.weight, 1, prune_out_dim_indices))
                    if not output_attentions:
                        attn_weights = None
                    
                    # --------------------------------------
                    #full inference with adding some info to layer input
                    bsz, q_len, _ = hidden_states.size()

                    # print('probe_out_dim_indices', probe_out_dim_indices)
                    # temp = torch.arange(4096).dtype(probe_out_dim_indices.dtype).to(probe_out_dim_indices.device)
                    # temp = torch.arange(4096).to(dtype=probe_out_dim_indices.dtype, device=probe_out_dim_indices.device)
                    if qk_prune_way is None:
                        probe_qk_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                        q_num_heads, k_num_heads = self.num_heads, self.num_key_value_heads
                        q_head_dim, k_head_dim = self.head_dim, self.head_dim
                    if vo_prune_way is None:
                        probe_vo_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                        v_num_heads, v_head_dim = self.num_heads, self.head_dim
                    # elif 'conditionqk' in cfg['prune_name']:
                    #     temp = probe_qk_out_dim_indices
                        # probe_out_dim_indices = torch.arange(4096).to(dtype=probe_out_dim_indices.dtype, device=probe_out_dim_indices.device)
                        # v_num_heads, v_head_dim = qk_num_heads, qk_head_dim
                    query_states = self.q_proj(hidden_states, probe_out_dim_indices=probe_qk_out_dim_indices)
                    key_states = self.k_proj(hidden_states, probe_out_dim_indices=probe_qk_out_dim_indices)
                    value_states = self.v_proj(hidden_states, probe_out_dim_indices=probe_vo_out_dim_indices)
                    # print('query_states shape', query_states.shape, flush=True)
                    # print('key_states shape', key_states.shape, flush=True)
                    # print('value_states shape', value_states.shape, flush=True)
                    # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                    # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                    # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                    query_states = query_states.view(bsz, q_len, q_num_heads, q_head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, k_num_heads, k_head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, v_num_heads, v_head_dim).transpose(1, 2)
                    # print('query_states shape 2', query_states.shape, flush=True)
                    # print('key_states shape 2', key_states.shape, flush=True)
                    # print('value_states shape 2', value_states.shape, flush=True)
                    kv_seq_len = key_states.shape[-2]
                    # print('past_key_value', past_key_value, flush=True)
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
                    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(k_head_dim)
                    print('query_states_fullinf', query_states.shape, flush=True)
                    print('attn_weights_fullinf', attn_weights.shape, flush=True)
                    if attn_weights.size() != (bsz, k_num_heads, q_len, kv_seq_len):
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
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    if 'delseq' in cfg['prune_method']:
                        print('delseq 2', flush=True)
                        # attn_weights = torch.index_select(attn_weights, -1, attn_weights_indices)
                        # value_states = torch.index_select(value_states, -2, attn_weights_indices)
                        # torch.cuda.reset_peak_memory_stats(device="cuda")  # Reset peak memory statistics
                        # before_op_memory_allocated = torch.cuda.memory_allocated("cuda")
                        # start_time = time.time()

                        attn_weights_indices_expand = attn_weights_indices.unsqueeze(0).unsqueeze(2).expand(bsz, k_num_heads, q_len, -1)
                        attn_weights = torch.gather(attn_weights, -1, attn_weights_indices_expand)
                        value_states_indices_expand = attn_weights_indices.unsqueeze(0).unsqueeze(3).expand(bsz, v_num_heads, -1, v_head_dim)
                        value_states = torch.gather(value_states, -2, value_states_indices_expand)

                        # end_time = time.time()
                        # elapsed_time = end_time - start_time
                        # print(f"Elapsed time: {elapsed_time} seconds")  
                        # after_op_memory_allocated = torch.cuda.memory_allocated("cuda")
                        # memory_used = after_op_memory_allocated - before_op_memory_allocated
                        # print(f"Memory used for the operation: {memory_used} bytes")

                    # value_states: bsz, self.num_key_value_heads, q_len, self.head_dim
                    # value_states: 1, self.num_key_value_heads, q_len, self.head_dim
                    attn_output = torch.matmul(attn_weights, value_states)
                    # print('attn_output after value', attn_output, attn_output.shape, flush=True)
                    if attn_output.size() != (bsz, v_num_heads, q_len, v_head_dim):
                        raise ValueError(
                            f"`attn_output` should be of size {(bsz, v_num_heads, q_len, v_head_dim)}, but is"
                            f" {attn_output.size()}"
                        )
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                    attn_output = attn_output.reshape(bsz, q_len, v_num_heads * v_head_dim)
                    # print('final attn input', attn_output, attn_output.shape, flush=True)
                    attn_output = self.o_proj(attn_output, probe_in_dim_indices=probe_vo_out_dim_indices)
                    # if 'restore' in cfg['prune_name']:
                    #     print('restore', flush=True)
                    #     restore = restore.expand(bsz, q_len, self.hidden_size)
                    #     attn_output = attn_output + restore
                    # print('attn_output_final', attn_output, attn_output.shape, flush=True)
                    if not output_attentions:
                        attn_weights = None
                    
                    return attn_output, attn_weights, past_key_value
                elif ('calib' in cfg['prune_method'] or 'runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']):
                    time_start = time.time()
                    bsz, q_len, _ = hidden_states.size()

                    q_num_heads, k_num_heads, v_num_heads = self.num_heads, self.num_key_value_heads, self.num_key_value_heads
                    q_head_dim, k_head_dim, v_head_dim = self.head_dim, self.head_dim, self.head_dim

                    qk_prune_way = cfg['qk_prune_way']
                    vo_prune_way = cfg['vo_prune_way']
                    
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
                                # probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_probe_mlp_metric(probe_out_dim_metric, multiple)
                                probe_vo_out_dim_indices, probe_vo_out_dim_indices_for_rope, v_num_heads, v_head_dim = self.pruning_module.sort_probe_attn_metric(probe_out_dim_metric, v_num_heads, v_head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])
                    
                    if qk_prune_way is None:
                        probe_qk_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                        probe_qk_out_dim_indices_for_rope = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                        # print('probe_qk_out_dim_indices', probe_qk_out_dim_indices, flush=True)
                    else:
                        if 'bv' in qk_prune_way:
                            qk_prune_way = vo_prune_way
                            q_num_heads, k_num_heads = v_num_heads, v_num_heads
                            q_head_dim, k_head_dim = v_head_dim, v_head_dim
                            probe_qk_out_dim_indices = probe_vo_out_dim_indices
                            probe_qk_out_dim_indices_for_rope = probe_vo_out_dim_indices_for_rope
                        
                        # print('qk_prune_way', qk_prune_way, flush=True)
                        # print('num_heads', q_num_heads, k_num_heads, flush=True)
                        # print('head_dim', q_head_dim, k_head_dim, flush=True)

                        # whole will prune neuron, no effect
                        # if 'noqk' in cfg['prune_method'] and qk_prune_way != 'whole':
                        #     qk_prune_way = None
                        #     q_num_heads, k_num_heads = self.num_heads, self.num_heads
                        #     q_head_dim, k_head_dim = self.head_dim, self.head_dim
                        #     probe_qk_out_dim_indices = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                        #     probe_qk_out_dim_indices_for_rope = torch.arange(self.hidden_size, dtype=torch.long).to(device=hidden_states.device)
                        
                        
                    query_states = self.q_proj(hidden_states, probe_out_dim_indices=probe_qk_out_dim_indices)
                    key_states = self.k_proj(hidden_states, probe_out_dim_indices=probe_qk_out_dim_indices)
                    value_states = self.v_proj(hidden_states, probe_out_dim_indices=probe_vo_out_dim_indices)

                    print('query_states shape', query_states.shape, flush=True)
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
                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

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

                    attn_output = self.o_proj(attn_output, probe_in_dim_indices=probe_vo_out_dim_indices)
                    if not output_attentions:
                        attn_weights = None

                    custom_duration = time.time() - time_start
                    # print('custom_duration llama attention', custom_duration, flush=True)
                    return attn_output, attn_weights, past_key_value
        else:
            time_start = time.time()
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

        start_time = time.time()
        residual = hidden_states
        temp_attn_residual = copy.deepcopy(residual)

        hidden_states = self.input_layernorm(hidden_states)

        kwargs['temp_mlp_residual'] = self.input_layernorm(kwargs['temp_mlp_residual'])
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        prev_hidden_states = copy.deepcopy(hidden_states).to(torch.float32)
        # print('prev_hidden_states', prev_hidden_states, prev_hidden_states.abs().sum(), flush=True)
        # print('prev residual', temp_attn_residual, temp_attn_residual.to(torch.float32).abs().sum(), flush=True)
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        temp_mlp_residual = copy.deepcopy(residual)
        

        # import numpy as np

        def calculate_sign_match_and_difference_percentage(hidden_states, temp):
            # Ensure hidden_states and temp are PyTorch tensors
            hidden_states = torch.tensor(hidden_states)
            temp = torch.tensor(temp)
            
            # Check if the signs match (True where signs match, False otherwise)
            sign_matches = torch.sign(hidden_states) == torch.sign(temp)
            
            # Count the number of sign matches
            sign_match_count = torch.sum(sign_matches).item()
            
            # Calculate the ratio of sign matches
            total_elements = hidden_states.numel()
            sign_match_ratio = sign_match_count / total_elements
            
            # Calculate the difference percentage for each position
            # Use torch.clamp to avoid division by zero, setting a minimum value
            hidden_states_nonzero = torch.clamp(hidden_states, min=1e-6, max=65504)
            temp_nonzero = torch.clamp(temp, min=1e-6, max=65504)
            difference_percentage = (torch.abs(temp_nonzero) / torch.abs(hidden_states_nonzero)) * 100
            
            # difference_sum = torch.sum(torch.abs(hidden_states - temp)) / hidden_states.numel()
            return sign_match_count, sign_match_ratio, difference_percentage

        flattened_hidden_states = hidden_states.flatten()
        num_elements_to_select = max(1, int(0.10 * flattened_hidden_states.numel()))  # Top 10% of elements
        # Select the top 10% elements based on their absolute value
        abs_flattened_hidden_states = flattened_hidden_states.abs()
        values, indices = torch.topk(abs_flattened_hidden_states, num_elements_to_select)

        ## Retrieve the actual values from the original tensor using these indices
        selected_values = flattened_hidden_states[indices].to(torch.float32)

        # Calculate the L1 norm (sum of absolute values) and L2 norm (square root of sum of squares) of these values
        l1_norm = selected_values.abs().sum()
        l2_norm = torch.sqrt((selected_values ** 2).sum())

        flattened_post_temp_attn_residual = temp_attn_residual.flatten()
        post_temp_attn_residual_values = flattened_post_temp_attn_residual[indices].to(torch.float32)


        l1_diff_norm = (selected_values - post_temp_attn_residual_values).abs().sum()
        l2_diff_norm = torch.sqrt(((selected_values - post_temp_attn_residual_values) ** 2).sum())

        sign_matches = torch.sign(selected_values) == torch.sign(post_temp_attn_residual_values)
        sign_match_ratio = torch.sum(sign_matches).item() / num_elements_to_select    

        cosine_similarity = torch.nn.functional.cosine_similarity(
            selected_values.float(),  # Ensure the data type is float for cosine similarity computation
            post_temp_attn_residual_values.float(),
            dim=0  # Compute the cosine similarity across the dimension 0 (element-wise for vectors)
        )


        # print('before l1_norm', l1_norm, flush=True)
        # print('before l2_norm', l2_norm, flush=True)
        # print('before l1_diff_norm', l1_diff_norm, flush=True)
        # print('before l2_diff_norm', l2_diff_norm, flush=True)
        # print('before sign_match_ratio', sign_match_ratio, flush=True)
        # print('before cosine_similarity', cosine_similarity, flush=True)


        hidden_states = self.post_attention_layernorm(hidden_states)
        post_temp_attn_residual = self.post_attention_layernorm(temp_attn_residual)

        flattened_hidden_states = hidden_states.flatten()
        num_elements_to_select = max(1, int(0.10 * flattened_hidden_states.numel()))  # Top 10% of elements
        # Select the top 10% elements based on their absolute value
        abs_flattened_hidden_states = flattened_hidden_states.abs()
        values, indices = torch.topk(abs_flattened_hidden_states, num_elements_to_select)

        ## Retrieve the actual values from the original tensor using these indices
        selected_values = flattened_hidden_states[indices].to(torch.float32)

        # Calculate the L1 norm (sum of absolute values) and L2 norm (square root of sum of squares) of these values
        l1_norm = selected_values.abs().sum()
        l2_norm = torch.sqrt((selected_values ** 2).sum())

        flattened_post_temp_attn_residual = post_temp_attn_residual.flatten()
        post_temp_attn_residual_values = flattened_post_temp_attn_residual[indices].to(torch.float32)


        l1_diff_norm = (selected_values - post_temp_attn_residual_values).abs().sum()
        l2_diff_norm = torch.sqrt(((selected_values - post_temp_attn_residual_values) ** 2).sum())

        sign_matches = torch.sign(selected_values) == torch.sign(post_temp_attn_residual_values)
        sign_match_ratio = torch.sum(sign_matches).item() / num_elements_to_select    

        cosine_similarity = torch.nn.functional.cosine_similarity(
            selected_values.float(),  # Ensure the data type is float for cosine similarity computation
            post_temp_attn_residual_values.float(),
            dim=0  # Compute the cosine similarity across the dimension 0 (element-wise for vectors)
        )

        # print('l1_norm', l1_norm, flush=True)
        # print('l2_norm', l2_norm, flush=True)
        # print('l1_diff_norm', l1_diff_norm, flush=True)
        # print('l2_diff_norm', l2_diff_norm, flush=True)
        # print('sign_match_ratio', sign_match_ratio, flush=True)
        # print('cosine_similarity', cosine_similarity, flush=True)
        # print('selected_values', selected_values, flush=True)
        # print('post_temp_attn_residual_values', post_temp_attn_residual_values, flush=True)



        # sign_match_count, sign_match_ratio, difference_percentage = calculate_sign_match_and_difference_percentage(hidden_states, post_residual)

        # print("Number of Sign Matches:", sign_match_count)
        # print("Ratio of Sign Matches:", sign_match_ratio)
        # print("difference_percentage for Each Position:\n", difference_percentage)

        # print('residual', residual)
        # print('hidden_states', hidden_states)
        # sorted_residual_values, _ = torch.sort(post_residual, dim=-1)
        # sorted_hidden_states_values, _ = torch.sort(hidden_states, dim=-1)

        # Set print options to ensure all values are printed if the tensors are not too large
        # torch.set_printoptions(threshold=5000)

        # # Print sorted values
        # print('Sorted post residual:', sorted_residual_values)
        # print('Sorted post hidden_states:', sorted_hidden_states_values)
        hidden_states = self.mlp(hidden_states, post_attn_residual=post_temp_attn_residual)

        
        hidden_states = residual + hidden_states
        # sorted_residual_values, _ = torch.sort(temp_mlp_residual, dim=-1)
        # sorted_hidden_states_values, _ = torch.sort(hidden_states, dim=-1)
        # print('Sorted post residual:', sorted_residual_values)
        # print('Sorted post hidden_states:', sorted_hidden_states_values)
        # sign_match_count, sign_match_ratio, difference_percentage = calculate_sign_match_and_difference_percentage(hidden_states, temp)

        # print("after mlp Number of Sign Matches:", sign_match_count)
        # print("after mlp Ratio of Sign Matches:", sign_match_ratio)
        # print("after mlp difference_percentage for Each Position:\n", difference_percentage)


        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        custom_duration = time.time() - start_time
        # print('custom_duration decoder layer', custom_duration, flush=True)
        return outputs, temp_mlp_residual


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
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_order) for layer_order in range(config.num_hidden_layers)])
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
        temp_mlp_residual = inputs_embeds

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
                layer_outputs, temp_mlp_residual = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    temp_mlp_residual=temp_mlp_residual
                )
            # print('layer_outputs', layer_outputs, flush=True)
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

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
            # print('shift_logits', shift_logits, shift_logits.shape, flush=True)
            # print('shift_labels', shift_labels, shift_labels.shape, flush=True)
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