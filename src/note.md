_prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
def _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window)

key_value_length = input_shape[-1] + past_key_values_length


attention_mask = attn_mask_converter.to_4d(attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype)

attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
def to_4d(attention_mask_2d, query_length, dtype, key_value_length)




_expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1])
def _expand_mask(mask, dtype, tgt_len):