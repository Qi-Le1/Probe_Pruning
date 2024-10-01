# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch OPT model."""
from typing import List, Optional, Tuple, Union

import time
import copy
import torch
import torch.utils.checkpoint
import traceback 
import threading
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import OPTConfig
from config import cfg
from ..pruning_module import HiddenRepresentationPruning
from .utils import generate_probe, get_next_layer  

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]
'''
Note: transformers 4.35.0 version
'''


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        layer_order: int = 0,
        config: dict = {}
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.layer_order = layer_order
        self.q_num_heads, self.k_num_heads, self.v_num_heads = self.num_heads, self.num_heads, self.num_heads
        self.pruning_module = HiddenRepresentationPruning(cfg, f'opt_attention_{layer_order}', config)


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, num_head=None):
        if num_head is None:
            return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        else:
            return tensor.view(bsz, seq_len, num_head, self.head_dim).transpose(1, 2).contiguous()

    def probe_process(
        self, 
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ):

        qk_prune_way = cfg['qk_prune_way']
        vo_prune_way = cfg['vo_prune_way']
        full_bsz, tgt_len, _ = hidden_states.size()
        if cfg['q_probe_ratio'] == cfg['k_probe_ratio'] and cfg['q_probe_ratio'] == cfg['v_probe_ratio']:
            if 'respick' in cfg['prune_method']:
                residual_for_probe = kwargs['respick']
            else:
                residual_for_probe = None
            probe, bsz_selected_indices, seq_selected_indices = generate_probe(hidden_states, cfg[f'q_probe_ratio'], residual_for_probe)
        else:
            raise ValueError('q_probe_num should be equal to k_probe_num and v_probe_num for now')

        bsz, src_len, _ = probe.size()
        self.q_num_heads, self.k_num_heads, self.v_num_heads = self.num_heads, self.num_heads, self.num_heads

        # get query proj
        query_states = self.q_proj(probe, cal_attn_probe_out_dim_metric=True) * self.scaling
        # self_attention
        key_states = self._shape(self.k_proj(probe, cal_attn_probe_out_dim_metric=True), -1, bsz)
        value_states = self._shape(self.v_proj(probe, cal_attn_probe_out_dim_metric=True), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, src_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, src_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, src_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (full_bsz, 1, tgt_len, tgt_len):
                raise ValueError(
                    f"Attention mask should be of size {(full_bsz, 1, tgt_len, tgt_len)}, but is {attention_mask.size()}"
                )
        
        if bsz_selected_indices is None:
            probe_attn_mask = attention_mask[:, :, :src_len, :src_len]
        else:
            probe_attn_mask = attention_mask[bsz_selected_indices, :, :src_len, :src_len]

        attn_weights = attn_weights.view(bsz, self.num_heads, src_len, src_len) + probe_attn_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, src_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        
        attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
            
        if attn_output.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, src_len, self.embed_dim)

        if 'calib' in cfg['prune_method'] or 'runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']:
            probe_out_dim_metric = self.pruning_module.cal_attn_prune_metric(attn_output, self.out_proj.weight.data, cfg['prune_metric'], bsz_selected_indices, seq_selected_indices, global_metric_score_distribution=self.out_proj.get_global_metric_score_distribution(tgt_len))
        else:
            probe_out_dim_metric = self.pruning_module.cal_attn_prune_metric(attn_output, self.out_proj.weight.data, cfg['prune_metric'], bsz_selected_indices, seq_selected_indices)

        if 'flapratio' in cfg['prune_method']:
            probe_vo_out_dim_indices, self.v_num_heads, _ = self.pruning_module.sort_attn_metric(probe_out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'], pruning_ratio=self.out_proj.pruning_ratio)
        else:
            probe_vo_out_dim_indices, self.v_num_heads, _ = self.pruning_module.sort_attn_metric(probe_out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])

        
        self.q_num_heads, self.k_num_heads = self.v_num_heads, self.v_num_heads
        probe_qk_out_dim_indices = probe_vo_out_dim_indices

        self.q_proj.prepare_async_weight(out_dim_indices=probe_qk_out_dim_indices)
        self.k_proj.prepare_async_weight(out_dim_indices=probe_qk_out_dim_indices)
        self.v_proj.prepare_async_weight(out_dim_indices=probe_vo_out_dim_indices)
        self.out_proj.prepare_async_weight(in_dim_indices=probe_vo_out_dim_indices)
        return probe_qk_out_dim_indices, probe_vo_out_dim_indices, attn_output, attn_weights_reshaped, past_key_value
    

    def attention_forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ):  
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        if cfg['pad_tokens'] is not None:
            cfg['pad_tokens'].to(attn_weights.device) 
            attn_output[cfg['pad_tokens']] = 0
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""


        if cfg['calibration_stage'] == True:
            return self.attention_forward(hidden_states, key_value_states, past_key_value, attention_mask, layer_head_mask, output_attentions, **kwargs)
        elif cfg['calibration_stage'] == False :
            if ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'out_proj' in cfg['cust_tgt_modules']) and self.layer_order not in cfg['skip_layers']:
                bsz, q_len, _ = hidden_states.size()
                probe_qk_out_dim_indices, probe_vo_out_dim_indices = None, None
                if 'probe' in cfg['prune_method']:
                    qk_prune_way = cfg['qk_prune_way']
                    vo_prune_way = cfg['vo_prune_way']
                    if cfg['mode'] == 'sync':
                        probe_qk_out_dim_indices, probe_vo_out_dim_indices, attn_weights, attn_weights_reshaped, past_key_value = self.probe_process(hidden_states, key_value_states, past_key_value, attention_mask, layer_head_mask, output_attentions, **kwargs)
                        # calculate probe's FLOPs
                        if cfg['onlyprobe'] == True:
                            attn_output = torch.zeros((cfg['batch_size'], hidden_states.shape[1], self.embed_dim), device=hidden_states.device, dtype=hidden_states.dtype)
                            return attn_output, None, past_key_value
                    elif cfg['mode'] == 'asyncintra':
                        # if not, do full inference
                        if 'input_layernorm_mlp_residual' in kwargs:
                            _, _, _, _, _ = self.probe_process(kwargs['input_layernorm_mlp_residual'], key_value_states, past_key_value, attention_mask, layer_head_mask, output_attentions, **kwargs)
                            # return
                        else:
                            pass
                            # raise ValueError('Invalid input for asyncintra mode')
                    
                    # --------------------------------------
                    is_cross_attention = key_value_states is not None
                    qk_prune_way = cfg['qk_prune_way']
                    vo_prune_way = cfg['vo_prune_way']
                    bsz, tgt_len, _ = hidden_states.size()

                    # get query proj
                    query_states = self.q_proj(hidden_states, out_dim_indices=probe_qk_out_dim_indices) * self.scaling
                    key_states = self._shape(self.k_proj(hidden_states, out_dim_indices=probe_qk_out_dim_indices), -1, bsz, self.q_num_heads)
                    value_states = self._shape(self.v_proj(hidden_states, out_dim_indices=probe_vo_out_dim_indices), -1, bsz, self.q_num_heads)

                    if self.is_decoder:
                        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
                        # Further calls to cross_attention layer can then reuse all cross-attention
                        # key/value_states (first "if" case)
                        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
                        # all previous decoder key/value_states. Further calls to uni-directional self-attention
                        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
                        # if encoder bi-directional self-attention `past_key_value` is always `None`
                        past_key_value = (key_states, value_states)

                    proj_shape = (bsz * self.q_num_heads, -1, self.head_dim)
                    query_states = self._shape(query_states, tgt_len, bsz, self.q_num_heads).view(*proj_shape)
                    key_states = key_states.view(*proj_shape)
                    value_states = value_states.view(*proj_shape)

                    src_len = key_states.size(1)
                    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

                    if attn_weights.size() != (bsz * self.q_num_heads, tgt_len, src_len):
                        raise ValueError(
                            f"Attention weights should be of size {(bsz * self.q_num_heads, tgt_len, src_len)}, but is"
                            f" {attn_weights.size()}"
                        )

                    if attention_mask is not None:
                        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                            raise ValueError(
                                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                            )
                        attn_weights = attn_weights.view(bsz, self.q_num_heads, tgt_len, src_len) + attention_mask
                        attn_weights = torch.max(
                            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                        )
                        attn_weights = attn_weights.view(bsz * self.q_num_heads, tgt_len, src_len)

                    # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
                    if attn_weights.dtype == torch.float16:
                        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
                    else:
                        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

                    if output_attentions:
                        # this operation is a bit awkward, but it's required to
                        # make sure that attn_weights keeps its gradient.
                        # In order to do so, attn_weights have to be reshaped
                        # twice and have to be reused in the following
                        attn_weights_reshaped = attn_weights.view(bsz, self.q_num_heads, tgt_len, src_len)
                        attn_weights = attn_weights_reshaped.view(bsz * self.q_num_heads, tgt_len, src_len)
                    else:
                        attn_weights_reshaped = None

                    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

                    attn_output = torch.bmm(attn_probs, value_states)

                    if attn_output.size() != (bsz * self.q_num_heads, tgt_len, self.head_dim):
                        raise ValueError(
                            f"`attn_output` should be of size {(bsz, self.q_num_heads, tgt_len, self.head_dim)}, but is"
                            f" {attn_output.size()}"
                        )

                    attn_output = attn_output.view(bsz, self.q_num_heads, tgt_len, self.head_dim)
                    attn_output = attn_output.transpose(1, 2)

                    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
                    # partitioned aross GPUs when using tensor-parallelism.
                    attn_output = attn_output.reshape(bsz, tgt_len, self.q_num_heads * self.head_dim)
                    if cfg['pad_tokens'] is not None:
                        cfg['pad_tokens'].to(attn_weights.device) 
                        attn_output[cfg['pad_tokens']] = 0
                    attn_output = self.out_proj(attn_output, in_dim_indices=probe_vo_out_dim_indices)
                    
                    return attn_output, attn_weights_reshaped, past_key_value
                elif 'calib' in cfg['prune_method'] and ('runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']):
                    # if key_value_states are provided this layer is used as a cross-attention layer
                    # for the decoder
                    is_cross_attention = key_value_states is not None

                    qk_prune_way = cfg['qk_prune_way']
                    vo_prune_way = cfg['vo_prune_way']
                    bsz, tgt_len, _ = hidden_states.size()

                    # get query proj
                    query_states = self.q_proj(hidden_states) * self.scaling
                    key_states = self._shape(self.k_proj(hidden_states), -1, bsz, self.q_num_heads)
                    value_states = self._shape(self.v_proj(hidden_states), -1, bsz, self.q_num_heads)

                    if self.is_decoder:
                        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
                        # Further calls to cross_attention layer can then reuse all cross-attention
                        # key/value_states (first "if" case)
                        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
                        # all previous decoder key/value_states. Further calls to uni-directional self-attention
                        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
                        # if encoder bi-directional self-attention `past_key_value` is always `None`
                        past_key_value = (key_states, value_states)

                    proj_shape = (bsz * self.q_num_heads, -1, self.head_dim)
                    query_states = self._shape(query_states, tgt_len, bsz, self.q_num_heads).view(*proj_shape)
                    key_states = key_states.view(*proj_shape)
                    value_states = value_states.view(*proj_shape)

                    src_len = key_states.size(1)
                    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

                    if attn_weights.size() != (bsz * self.q_num_heads, tgt_len, src_len):
                        raise ValueError(
                            f"Attention weights should be of size {(bsz * self.q_num_heads, tgt_len, src_len)}, but is"
                            f" {attn_weights.size()}"
                        )

                    if attention_mask is not None:
                        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                            raise ValueError(
                                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                            )
                        attn_weights = attn_weights.view(bsz, self.q_num_heads, tgt_len, src_len) + attention_mask
                        attn_weights = torch.max(
                            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                        )
                        attn_weights = attn_weights.view(bsz * self.q_num_heads, tgt_len, src_len)

                    # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
                    if attn_weights.dtype == torch.float16:
                        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
                    else:
                        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

                    if output_attentions:
                        # this operation is a bit awkward, but it's required to
                        # make sure that attn_weights keeps its gradient.
                        # In order to do so, attn_weights have to be reshaped
                        # twice and have to be reused in the following
                        attn_weights_reshaped = attn_weights.view(bsz, self.q_num_heads, tgt_len, src_len)
                        attn_weights = attn_weights_reshaped.view(bsz * self.q_num_heads, tgt_len, src_len)
                    else:
                        attn_weights_reshaped = None

                    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

                    attn_output = torch.bmm(attn_probs, value_states)

                    if attn_output.size() != (bsz * self.q_num_heads, tgt_len, self.head_dim):
                        raise ValueError(
                            f"`attn_output` should be of size {(bsz, self.q_num_heads, tgt_len, self.head_dim)}, but is"
                            f" {attn_output.size()}"
                        )

                    attn_output = attn_output.view(bsz, self.q_num_heads, tgt_len, self.head_dim)
                    attn_output = attn_output.transpose(1, 2)

                    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
                    # partitioned aross GPUs when using tensor-parallelism.
                    attn_output = attn_output.reshape(bsz, tgt_len, self.q_num_heads * self.head_dim)

                    attn_output = self.out_proj(attn_output)
                    if cfg['mode'] == 'asyncinter':
                        # with torch.cuda.stream(cfg['cuda_stream1']):
                        if torch.all(self.out_proj.get_global_metric_score_distribution() == 0):
                            vo_out_dim_indices = torch.arange(self.embed_dim, dtype=torch.long).to(device=hidden_states.device)
                        else:
                            out_dim_metric = self.pruning_module.cal_attn_calib_prune_metric(self.out_proj.get_global_metric_score_distribution(), self.out_proj.weight.data, cfg['prune_metric'])
                            if 'flapratio' in cfg['prune_method']:
                                vo_out_dim_indices, self.v_num_heads, _ = self.pruning_module.sort_attn_metric(out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'], pruning_ratio=self.out_proj.pruning_ratio)
                            else:
                                vo_out_dim_indices, self.v_num_heads, _ = self.pruning_module.sort_attn_metric(out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])

                        self.q_num_heads, self.k_num_heads = self.v_num_heads, self.v_num_heads
                        qk_out_dim_indices = vo_out_dim_indices
                                
                        self.q_proj.prepare_async_weight(out_dim_indices=qk_out_dim_indices)
                        self.k_proj.prepare_async_weight(out_dim_indices=qk_out_dim_indices)
                        self.v_proj.prepare_async_weight(out_dim_indices=vo_out_dim_indices)
                        self.out_proj.prepare_async_weight(in_dim_indices=vo_out_dim_indices)
                    else:
                        raise ValueError('please use asyncinter for calib+ema')
                    return attn_output, attn_weights_reshaped, past_key_value                
                elif 'calib' in cfg['prune_method'] or 'flap' in cfg['prune_method'] or 'wandasp' in cfg['prune_method']:
                    if cfg['mode'] == 'asyncinter':
                        # if key_value_states are provided this layer is used as a cross-attention layer
                        # for the decoder
                        is_cross_attention = key_value_states is not None

                        qk_prune_way = cfg['qk_prune_way']
                        vo_prune_way = cfg['vo_prune_way']
                        bsz, tgt_len, _ = hidden_states.size()

                        # get query proj
                        query_states = self.q_proj(hidden_states) * self.scaling
                        key_states = self._shape(self.k_proj(hidden_states), -1, bsz, self.q_num_heads)
                        value_states = self._shape(self.v_proj(hidden_states), -1, bsz, self.q_num_heads)

                        if self.is_decoder:
                            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
                            # Further calls to cross_attention layer can then reuse all cross-attention
                            # key/value_states (first "if" case)
                            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
                            # all previous decoder key/value_states. Further calls to uni-directional self-attention
                            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
                            # if encoder bi-directional self-attention `past_key_value` is always `None`
                            past_key_value = (key_states, value_states)

                        proj_shape = (bsz * self.q_num_heads, -1, self.head_dim)
                        # print('proj_shape', proj_shape, bsz, self.q_num_heads, self.head_dim)
                        # print('query_states', query_states.shape)
                        query_states = self._shape(query_states, tgt_len, bsz, self.q_num_heads).view(*proj_shape)
                        key_states = key_states.view(*proj_shape)
                        value_states = value_states.view(*proj_shape)

                        src_len = key_states.size(1)
                        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

                        if attn_weights.size() != (bsz * self.q_num_heads, tgt_len, src_len):
                            raise ValueError(
                                f"Attention weights should be of size {(bsz * self.q_num_heads, tgt_len, src_len)}, but is"
                                f" {attn_weights.size()}"
                            )

                        if attention_mask is not None:
                            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                                raise ValueError(
                                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                                )
                            attn_weights = attn_weights.view(bsz, self.q_num_heads, tgt_len, src_len) + attention_mask
                            attn_weights = torch.max(
                                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                            )
                            attn_weights = attn_weights.view(bsz * self.q_num_heads, tgt_len, src_len)

                        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
                        if attn_weights.dtype == torch.float16:
                            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
                        else:
                            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

                        if output_attentions:
                            # this operation is a bit awkward, but it's required to
                            # make sure that attn_weights keeps its gradient.
                            # In order to do so, attn_weights have to be reshaped
                            # twice and have to be reused in the following
                            attn_weights_reshaped = attn_weights.view(bsz, self.q_num_heads, tgt_len, src_len)
                            attn_weights = attn_weights_reshaped.view(bsz * self.q_num_heads, tgt_len, src_len)
                        else:
                            attn_weights_reshaped = None

                        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

                        attn_output = torch.bmm(attn_probs, value_states)

                        if attn_output.size() != (bsz * self.q_num_heads, tgt_len, self.head_dim):
                            raise ValueError(
                                f"`attn_output` should be of size {(bsz, self.q_num_heads, tgt_len, self.head_dim)}, but is"
                                f" {attn_output.size()}"
                            )

                        attn_output = attn_output.view(bsz, self.q_num_heads, tgt_len, self.head_dim)
                        attn_output = attn_output.transpose(1, 2)

                        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
                        # partitioned aross GPUs when using tensor-parallelism.
                        attn_output = attn_output.reshape(bsz, tgt_len, self.q_num_heads * self.head_dim)

                        attn_output = self.out_proj(attn_output)

                        if cfg['cur_batch_index'] == 0:
                            if torch.all(self.out_proj.get_global_metric_score_distribution() == 0):
                                print('jinzheli1')
                                vo_out_dim_indices = torch.arange(self.embed_dim, dtype=torch.long).to(device=hidden_states.device)
                            else:
                                print('jinzheli2')
                                out_dim_metric = self.pruning_module.cal_attn_calib_prune_metric(self.out_proj.get_global_metric_score_distribution(), self.out_proj.weight.data, cfg['prune_metric'])
                                if 'flapratio' in cfg['prune_method']:
                                    vo_out_dim_indices, self.v_num_heads, _ = self.pruning_module.sort_attn_metric(out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'], pruning_ratio=self.out_proj.pruning_ratio)
                                else:
                                    vo_out_dim_indices, self.v_num_heads, _ = self.pruning_module.sort_attn_metric(out_dim_metric, self.num_heads, self.head_dim, vo_prune_way, 'vo', cfg['tc_multiple'])

                            self.q_num_heads, self.k_num_heads = self.v_num_heads, self.v_num_heads
                            qk_out_dim_indices = vo_out_dim_indices
                                    
                            self.q_proj.prepare_async_weight(out_dim_indices=qk_out_dim_indices)
                            self.k_proj.prepare_async_weight(out_dim_indices=qk_out_dim_indices)
                            self.v_proj.prepare_async_weight(out_dim_indices=vo_out_dim_indices)
                            self.out_proj.prepare_async_weight(in_dim_indices=vo_out_dim_indices)
                        return attn_output, attn_weights_reshaped, past_key_value
                    else:
                        raise ValueError('Invalid mode')
            else:
                return self.attention_forward(hidden_states, key_value_states, past_key_value, attention_mask, layer_head_mask, output_attentions, **kwargs)
        


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, layer_order):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.ffn_dim = config.ffn_dim
        self.optconfig = config

        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            bias=config.enable_bias,
            layer_order=layer_order,
            config=config
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

        self.cur_attn_inference_duration = 0
        self.cur_mlp_inference_duration = 0

        self.start_event = torch.cuda.Event(enable_timing=True, blocking=False)
        self.end_event = torch.cuda.Event(enable_timing=True, blocking=False)

        self.layer_order = layer_order
        self.pruning_module = HiddenRepresentationPruning(cfg, f'opt_mlp_{layer_order}', config)

        self.probe_out_dim_indices = None
        

    def probe_process(self, x, **kwargs):
        # 1. generate probeW
        # 2. run matrix multiplication
        # 3. calculate score
        # 4. extract metric

        # generate probe
        # rank / mean / absnml
        cur_batch_seq_len = x.size(1)
        
        if 'respick' in cfg['prune_method']:
            residual_for_probe = kwargs['respick']
        else:
            residual_for_probe = None
        probe, bsz_selected_indices, seq_selected_indices = generate_probe(x, cfg['fc1_probe_ratio'], residual_for_probe)
        
        probe_out = self.activation_fn(self.fc1(probe, cal_mlp_probe_out_dim_metric=True))
        # print('probe_out', probe_out.shape)
        # calculate score
        if 'calib' in cfg['prune_method'] or 'runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']:
            probe_out_dim_metric = self.pruning_module.cal_mlp_prune_metric(probe_out, self.fc2.weight.data, cfg['prune_metric'], bsz_selected_indices, seq_selected_indices, global_metric_score_distribution=self.fc2.get_global_metric_score_distribution(cur_batch_seq_len))
        else:
            probe_out_dim_metric = self.pruning_module.cal_mlp_prune_metric(probe_out, self.fc2.weight.data, cfg['prune_metric'], bsz_selected_indices, seq_selected_indices)

        if 'flapratio' in cfg['prune_method']:
            probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.fc2.pruning_ratio)
        else:
            probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, cfg['tc_multiple'])

        # extract matrix
        self.fc1.prepare_async_weight(out_dim_indices=probe_out_dim_indices)
        self.fc2.prepare_async_weight(in_dim_indices=probe_out_dim_indices)
        return probe_out_dim_indices, probe_out
    
    def mlp_layer(self, hidden_states, **kwargs):
        if cfg['calibration_stage'] == True:
            hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))         
            return hidden_states  
        elif cfg['calibration_stage'] == False:
            if ('fc1' in cfg['cust_tgt_modules'] or 'fc2' in cfg['cust_tgt_modules']) and self.layer_order not in cfg['skip_layers']:
                if 'probe' in cfg['prune_method']:
                    probe_out_dim_indices = None
                    if cfg['mode'] == 'sync':
                        probe_out_dim_indices, probe_out = self.probe_process(hidden_states, **kwargs)
                        # count flops for probe
                        if cfg['onlyprobe'] == True:
                            # match the shape, and will not count the flops for this part
                            hidden_states = torch.zeros((cfg['batch_size'], hidden_states.shape[1], self.embed_dim), device=hidden_states.device, dtype=hidden_states.dtype)
                            return hidden_states
                    elif cfg['mode'] == 'asyncintra':
                        if 'post_layernorm_attn_residual' in kwargs:
                            _, _ = self.probe_process(kwargs['post_layernorm_attn_residual'], **kwargs)
                            # return
                        else:
                            pass
                            # raise ValueError('Invalid input for asyncintra mode')
                        
                    hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states, out_dim_indices=probe_out_dim_indices)), in_dim_indices=probe_out_dim_indices) 
                    return hidden_states
                elif 'calib' in cfg['prune_method'] and ('runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']):

                    if cfg['mode'] == 'asyncinter':
                        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))  

                        # with torch.cuda.stream(cfg['cuda_stream1']):
                        if torch.all(self.fc2.get_global_metric_score_distribution() == 0):
                            out_dim_indices = torch.arange(self.ffn_dim, dtype=torch.long).to(device=hidden_states.device)
                        else:
                            out_dim_metric = self.pruning_module.cal_mlp_calib_prune_metric(self.fc2.get_global_metric_score_distribution(), self.fc2.weight.data, cfg['prune_metric'])

                            if 'flapratio' in cfg['prune_method']:
                                out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.fc2.pruning_ratio)
                            else:
                                out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'])

                        self.fc1.prepare_async_weight(out_dim_indices=out_dim_indices)
                        self.fc2.prepare_async_weight(in_dim_indices=out_dim_indices)
                    else:
                        raise ValueError('Invalid mode')
                    return hidden_states
                # only calib (baselines)
                elif 'calib' in cfg['prune_method'] or 'flap' in cfg['prune_method'] or 'wandasp' in cfg['prune_method']:
                    # no ema or runningmean
                    if cfg['mode'] == 'asyncinter':
                        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))  

                        if cfg['cur_batch_index'] == 0:
                            if torch.all(self.fc2.get_global_metric_score_distribution() == 0):
                                out_dim_indices = torch.arange(self.ffn_dim, dtype=torch.long).to(device=hidden_states.device)
                            else:
                                out_dim_metric = self.pruning_module.cal_mlp_calib_prune_metric(self.fc2.get_global_metric_score_distribution(), self.fc2.weight.data, cfg['prune_metric'])

                                if 'flapratio' in cfg['prune_method'] or 'gridratio' in cfg['prune_method']:
                                    out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.fc2.pruning_ratio)
                                else:
                                    out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(out_dim_metric, cfg['tc_multiple'])

                            self.fc1.prepare_async_weight(out_dim_indices=out_dim_indices)
                            print('out_dim_indices', out_dim_indices.shape, self.fc2.weight.shape, self.fc2.bias.shape)
                            self.fc2.prepare_async_weight(in_dim_indices=out_dim_indices)
                    else:
                        raise ValueError('Invalid mode')
                    return hidden_states
            else:
                hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states))) 
                return hidden_states  

    def check_asyncintra_for_next_mlp(self):
        if ('fc1' in cfg['cust_tgt_modules'] or 'fc2' in cfg['cust_tgt_modules']) \
            and self.layer_order not in cfg['skip_layers'] \
            and cfg['calibration_stage'] == False \
            and cfg['mode'] == 'asyncintra' \
            and 'probe' in cfg['prune_method']:
            return True
        return False
    
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
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # cur_gpu = self.fc1.weight.device
        # with torch.cuda.device(cur_gpu):
        residual = hidden_states

        torch.cuda.synchronize()
        self.start_event.record()
        hidden_states = self.self_attn_layer_norm(hidden_states)
        if self.check_asyncintra_for_next_attention():
            input_layernorm_mlp_residual = self.self_attn_layer_norm(kwargs['last_layer_residual'])
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            respick=kwargs['last_layer_residual'],
            input_layernorm_mlp_residual=input_layernorm_mlp_residual
        )
        else:
            input_layernorm_mlp_residual = None
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                respick=residual
            )
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states


        # Fully Connected
        # hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        if self.check_asyncintra_for_next_mlp():
            post_layernorm_attn_residual = self.final_layer_norm(residual)
            respick = residual
        else:
            post_layernorm_attn_residual = None

        residual = hidden_states

        self.end_event.record()
        torch.cuda.synchronize()
        if hasattr(self, 'cur_attn_inference_duration') and cfg['cur_batch_index'] >= 1:
            self.cur_attn_inference_duration += self.start_event.elapsed_time(self.end_event)

        self.start_event.record()

        hidden_states = self.final_layer_norm(hidden_states)

        if self.check_asyncintra_for_next_mlp():
            hidden_states = self.mlp_layer(hidden_states, respick=respick, post_layernorm_attn_residual=post_layernorm_attn_residual)
        else:
            hidden_states = self.mlp_layer(hidden_states, respick=residual)

        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        self.end_event.record()
        torch.cuda.synchronize()
        if hasattr(self, 'cur_mlp_inference_duration') and cfg['cur_batch_index'] >= 1:
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


OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


OPT_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
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


class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config, layer_order) for layer_order in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds
        hidden_states[attention_mask == 0] = 0
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

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        last_layer_residual = None
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                next_layer = get_next_layer(self.layers, idx)
                layer_outputs, last_layer_residual = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    next_layer=next_layer,
                    last_layer_residual=last_layer_residual
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

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


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
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

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
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
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print('attention_mask_clm', attention_mask, attention_mask.shape, flush=True)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

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

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
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
    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPT_START_DOCSTRING,
)
class OPTForSequenceClassification(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
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
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
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

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value


@add_start_docstrings(
    """
    The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    OPT_START_DOCSTRING,
)
class OPTForQuestionAnswering(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value