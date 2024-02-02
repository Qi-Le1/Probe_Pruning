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
from ..pruning_module import HiddenRepresentationPruning
from module import nearest_even_number
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
    def __init__(self, config):
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

        self.cal_total_flops = True
        self.pruning_module = HiddenRepresentationPruning(cfg, 'llama_mlp')

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
            if 'probe' in cfg['prune_name'] and ('down_proj' in cfg['cust_tgt_modules'] or 'up_proj' in cfg['cust_tgt_modules'] or 'gate_proj' in cfg['cust_tgt_modules']):
                
                if 'mb' in cfg['prune_metric']:
                    x_mean_bsz = x.mean(axis=0)
                    comp_across_bsz_seq = torch.linalg.vector_norm(x_mean_bsz, ord=2, dim=0).unsqueeze_(0)
                elif 'mbms' in cfg['prune_metric']:
                    comp_across_bsz_seq = x.mean(axis=(0, 1)).unsqueeze_(0)
                else:
                    comp_across_bsz_seq = torch.linalg.vector_norm(x, ord=2, dim=(0,1)).unsqueeze_(0)

                probe_out = self.act_fn(self.gate_proj(comp_across_bsz_seq, cal_mlp_probe_out_dim_metric=True)) * self.up_proj(comp_across_bsz_seq, cal_mlp_probe_out_dim_metric=True)

                # ablation study for different metrics
                if 'wanda' in cfg['prune_metric']:
                    probe_out_dim_metric = (torch.linalg.vector_norm(probe_out, ord=2, dim=1).reshape((1, -1)) * torch.abs(self.down_proj.weight.data)).sum(axis=0)
                elif 'flap' in cfg['prune_metric']:
                    pass
                elif 'probe' in cfg['prune_metric']:
                    probe_out_dim_metric = torch.sqrt((torch.sum(torch.pow(probe_out, 2), dim=1).reshape((1, -1)) * torch.pow(self.down_proj.weight.data, 2)).sum(axis=0))

                probe_out_dim_indices = self.pruning_module.cal_probe_mlp_metric(probe_out_dim_metric)
                kwargs['probe_out_dim_indices'] = probe_out_dim_indices
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x, probe_out_dim_indices=probe_out_dim_indices)) * self.up_proj(x, probe_out_dim_indices=probe_out_dim_indices), **kwargs)
                return down_proj
            else:
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

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

    def __init__(self, config: LlamaConfig):
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

        self.cal_total_flops = True
        # self.default_num_heads = config.num_attention_heads
        # self.default_head_dim = self.hidden_size // self.default_num_heads
        # self.default_num_key_value_heads = config.num_key_value_heads

        self.pruning_module = HiddenRepresentationPruning(cfg, 'llama_attention')

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

        if 'probe' in cfg['prune_name'] and ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']):

            if 'mb' in cfg['prune_metric']:
                x_mean_bsz = x.mean(axis=0)
                comp_across_bsz_seq = torch.linalg.vector_norm(x_mean_bsz, ord=2, dim=0).unsqueeze_(0)
            else:
                comp_across_bsz_seq = torch.linalg.vector_norm(x, ord=2, dim=(0,1)).unsqueeze_(0)

            q_num_heads, k_num_heads, v_num_heads = self.num_heads, self.num_key_value_heads, self.num_key_value_heads
            q_head_dim, k_head_dim, v_head_dim = self.head_dim, self.head_dim, self.head_dim
            # print('hidden_states', hidden_states)
            # print('comp_across_bsz', comp_across_bsz, flush=True)
            # if 'para' in cfg['prune_name']:
            # copy orignal code and modify a little bit for forecast pruning
            # currently does not implement for group attention, but it should work too
            bsz, q_len, _ = comp_across_bsz.size()
            
            query_states = self.q_proj(comp_across_bsz, cal_attn_probe_out_dim_metric='q_proj' in cfg['cust_tgt_modules'])
            key_states = self.k_proj(comp_across_bsz, cal_attn_probe_out_dim_metric='k_proj' in cfg['cust_tgt_modules'])
            value_states = self.v_proj(comp_across_bsz, cal_attn_probe_out_dim_metric='v_proj' in cfg['cust_tgt_modules'])

            # only consider qk effect, v and o remain the same
            # if 'onlyqk' in cfg['prune_name']:
            if 'wanda' in cfg['prune_metric']:
                probe_qk_out_dim_metric = (torch.linalg.vector_norm(query_states, ord=2, dim=(0, 1)).reshape((1, 1, -1)) * torch.abs(key_states)).sum(axis=(0, 1))
            elif 'flap' in cfg['prune_metric']:
                pass
            elif 'probe' in cfg['prune_metric']:
                probe_qk_out_dim_metric = torch.sqrt((torch.sum(torch.pow(query_states, 2), dim=(0, 1)).reshape((1, 1, -1)) * torch.pow(key_states, 2)).sum(axis=(0, 1)))

            probe_qk_out_dim_indices, probe_qk_out_dim_indices_for_rope, qk_num_heads, qk_head_dim  = self.pruning_module.cal_probe_attn_metric(probe_qk_out_dim_metric, self.num_heads, self.head_dim, cfg['qk_proj_prune'])
            q_num_heads, k_num_heads = qk_num_heads, qk_num_heads
            q_head_dim, k_head_dim = qk_head_dim, qk_head_dim

            if 'conditionqk' in cfg['prune_name']:
                # query_states = self.q_proj(hidden_states, probe_out_dim_indices=probe_qk_out_dim_indices)
                # key_states = self.k_proj(hidden_states, probe_out_dim_indices=probe_qk_out_dim_indices)
                if 'fill' in cfg['prune_name']:
                    mask = torch.zeros(query_states.shape[-1], dtype=torch.bool, device=query_states.device)
                    mask[~probe_qk_out_dim_indices] = True
                    query_states[..., mask] = 0
                    key_states[..., mask] = 0
                else:
                    query_states = torch.index_select(query_states, -1, probe_qk_out_dim_indices)
                    key_states = torch.index_select(key_states, -1, probe_qk_out_dim_indices)
                    
                query_states = query_states.view(bsz, q_len, q_num_heads, q_head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, k_num_heads, k_head_dim).transpose(1, 2)
            else:
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            # if past_key_value is not None:
            #     kv_seq_len += past_key_value[0].shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            if 'each' in cfg['qk_proj_prune']:
                query_states, key_states = apply_rotary_pos_emb_for_prune_each_head(query_states, key_states, cos, sin, position_ids, probe_qk_out_dim_indices_for_rope)
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # key_states: bsz, self.num_key_value_heads, q_len, self.head_dim -> bsz, self.num_key_value_heads, self.head_dim, q_len
            if 'conditionqk' in cfg['prune_name']:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(k_head_dim)
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            # print('attn_weights', attn_weights.shape, query_states.shape,flush=True)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            probe_attn_mask = attention_mask[0].unsqueeze_(0)
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
                if probe_attn_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {probe_attn_mask.size()}"
                    )
                attn_weights = attn_weights + probe_attn_mask
                # print('attnweightsaftermask', attn_weights, attn_weights.shape, flush=True)

            # upcast attention to fp32
            # attn_weights: bsz, self.num_heads, q_len, q_len
            # attn_weights: 1, self.num_heads, q_len, q_len
            # value_states: 1, self.num_key_value_heads, q_len, self.head_dim
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if 'delseq' in cfg['prune_name']:
                if 'wanda' in cfg['prune_metric']:
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
                _, attn_weights_indices, _, _ = self.pruning_module.cal_probe_attn_weights_metric(attn_weights_metric, attn_weights.shape[1], 1, cfg['vo_proj_prune'])

            # print('attn_weightsafter softmax', attn_weights, attn_weights.shape, flush=True)
            # value_states: bsz, self.num_key_value_heads, q_len, self.head_dim
            # value_states: 1, self.num_key_value_heads, q_len, self.head_dim
            attn_output = torch.matmul(attn_weights, value_states)
            
            # print('attn_output00', attn_output[0][0], attn_output[0][0].shape, flush=True)
            if attn_output.size() != (bsz, v_num_heads, q_len, v_head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, v_num_heads, q_len, v_head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            # attn_output: bsz, self.num_heads, q_len, self.head_dim -> bsz, q_len, self.num_heads, self.head_dim
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            
            if 'wanda' in cfg['prune_metric']:
                probe_out_dim_metric = (torch.linalg.vector_norm(attn_output, ord=2, dim=(0, 1)).reshape((1, -1)) * torch.abs(self.o_proj.weight.data)).sum(axis=0)
            elif 'flap' in cfg['prune_metric']:
                pass
            elif 'probe' in cfg['prune_metric']:
                probe_out_dim_metric = torch.sqrt((torch.sum(torch.pow(attn_output, 2), dim=(0, 1)).reshape((1, -1)) * torch.pow(self.o_proj.weight.data, 2)).sum(axis=0))
            probe_out_dim_indices, _, v_num_heads, v_head_dim = self.pruning_module.cal_probe_attn_metric(probe_out_dim_metric, v_num_heads, v_head_dim, cfg['vo_proj_prune'])

            if not output_attentions:
                attn_weights = None
            
            # --------------------------------------
            #full inference with adding some info to layer input
            bsz, q_len, _ = hidden_states.size()

            # print('probe_out_dim_indices', probe_out_dim_indices)
            # temp = torch.arange(4096).dtype(probe_out_dim_indices.dtype).to(probe_out_dim_indices.device)
            # temp = torch.arange(4096).to(dtype=probe_out_dim_indices.dtype, device=probe_out_dim_indices.device)
            if 'onlyqk' in cfg['prune_name']:
                probe_out_dim_indices = torch.arange(4096).to(dtype=probe_out_dim_indices.dtype, device=probe_out_dim_indices.device)
                v_num_heads, v_head_dim = self.num_heads, self.head_dim
            if 'onlyvo' in cfg['prune_name']:
                probe_qk_out_dim_indices = torch.arange(4096).to(dtype=probe_out_dim_indices.dtype, device=probe_out_dim_indices.device)
                q_num_heads, k_num_heads = self.num_heads, self.num_key_value_heads
                q_head_dim, k_head_dim = self.head_dim, self.head_dim
            # elif 'conditionqk' in cfg['prune_name']:
            #     temp = probe_qk_out_dim_indices
                # probe_out_dim_indices = torch.arange(4096).to(dtype=probe_out_dim_indices.dtype, device=probe_out_dim_indices.device)
                # v_num_heads, v_head_dim = qk_num_heads, qk_head_dim
            query_states = self.q_proj(hidden_states, probe_out_dim_indices=probe_qk_out_dim_indices)
            key_states = self.k_proj(hidden_states, probe_out_dim_indices=probe_qk_out_dim_indices)
            value_states = self.v_proj(hidden_states, probe_out_dim_indices=probe_out_dim_indices)
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

            if 'each' in cfg['qk_proj_prune']:
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

            # upcast attention to fp32
            # attn_weights: bsz, self.num_heads, q_len, q_len
            # attn_weights: 1, self.num_heads, q_len, q_len
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if 'deleteseq' in cfg['prune_name']:
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
            attn_output = self.o_proj(attn_output, probe_out_dim_indices=probe_out_dim_indices)

            # print('attn_output_final', attn_output, attn_output.shape, flush=True)
            if not output_attentions:
                attn_weights = None
        else:
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

        return attn_output, attn_weights, past_key_value




class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            LlamaAttention(config=config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else LlamaFlashAttention2(config=config)
        )
        self.mlp = LlamaMLP(config)
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

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

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
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual=residual)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

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
    _skip_keys_device_placement = "past_key_values"
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
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
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
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

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
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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