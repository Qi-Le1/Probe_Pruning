import math
import torch
from config import cfg

def custom_expand_mask(mask, dtype, tgt_len=None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        expanded_mask = expanded_mask.transpose(-1, -2)
        # expanded_mask = mask.unsqueeze(2).repeat(1, 1, tgt_len) 

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), -1/src_len)

def rank_process(x, probe_num, probe_type, residual):
    print('rank_process', flush=True)
    print('x_size', x.size(), flush=True)
    # print('residual_size', residual.size(), flush=True)

    if 'randomrank' in cfg['prune_method']:
        print('randomrank', flush=True)
        if 'bsz' in probe_type:
            sorted_indices = torch.randperm(x.size(0))[:probe_num]
            x = x[sorted_indices, :, :]
        elif 'seq' in probe_type:
            sorted_indices = torch.randperm(x.size(1))[:probe_num]
            x = x[:, sorted_indices, :]
        return x, residual, sorted_indices
    elif 'rulerank' in cfg['prune_method']:
        cfg['temp_input_ids'] = cfg['temp_input_ids'].to(x.device)
        print('rulerank', flush=True)
        if 'bsz' in probe_type:
            # TODO: sort
            sorted_indices = torch.randperm(x.size(0))[:probe_num]
            # sorted_indices = torch.arange(probe_num)
            x = x[sorted_indices, :, :]
            cfg['temp_input_ids'] = cfg['temp_input_ids'][sorted_indices, :]
            if residual is not None:
                residual = residual[sorted_indices, :, :]
        elif 'seq' in probe_type:
            first_indices = torch.arange(1)

            # Calculate the start index for the last elements
            # start_index_from_end = max(4, last_dim_length - num_elements_from_end)  # Ensure there's no overlap with the first 4 indices

            # Generate indices for the last elements
            # probe_num = 1

            # cfg['num_nonpad_tokens'] = cfg['num_nonpad_tokens'].to(x.device)
            # probe_num = torch.tensor(probe_num).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)
            # a = torch.max(cfg['num_nonpad_tokens'] - probe_num, 1)[0].unsqueeze(1)
            # print('aaaaa', a)
            # last_indices = torch.arange(torch.max(cfg['num_nonpad_tokens'] - probe_num, 1)[0], cfg['num_nonpad_tokens'])
            if 'last' in cfg['prune_method']:
                last_indices = torch.arange(x.size(1) - probe_num + 1, x.size(1))
            else:
                last_indices = torch.arange(max(x.size(1) - probe_num - 1, 1), x.size(1))
                last_indices = torch.arange(1, probe_num)
            # last_indices = torch.arange(30)

            # Combine the two sets of indices
            sorted_indices = torch.cat((first_indices, last_indices)).to(x.device)
            # print('sorted_indices', sorted_indices, sorted_indices.shape, x.shape, flush=True)
            # sorted_indices = last_indices.to(x.device)
            # sorted_indices = first_indices.to(x.device)


            # if 'last' in cfg['prune_method']:
            #     last_indices = torch.arange(x.size(1) - probe_num, x.size(1))
            # else:
            #     last_indices = torch.arange(max(x.size(1) - probe_num - 1, 1), x.size(1))

            # l2_norms = torch.linalg.vector_norm(residual, ord=2, dim=2)
            # massive_tokens = torch.where(l2_norms > 500)
            # unique_batches, counts = torch.unique_consecutive(massive_tokens[0], return_counts=True)
            # result_tensors = torch.split(massive_tokens[1], counts.tolist())
            # print('massive_tokens', massive_tokens, flush=True)
            # # Stack the indices to create a 2D tensor of index pairs
            # if (l2_norms > 500).any():
            #     massive_tokens_index_pairs = torch.stack(result_tensors, dim=0).to(x.device)
            #     print('massive_tokens_index_pairs', massive_tokens_index_pairs, flush=True)
            #     last_indices = last_indices.repeat(x.size(0), 1).to(x.device)
            #     print('last_indices', last_indices, flush=True)
            #     select_indices = torch.cat((last_indices, massive_tokens_index_pairs), dim=1)
            # else:
            #     select_indices = last_indices.repeat(x.size(0), 1).to(x.device)


            # sorted_indices = torch.sort(select_indices, dim=1)[0]
            # sorted_indices = torch.stack([torch.unique(row) for row in sorted_indices])
            # batch_indices = torch.arange(x.size(0)).unsqueeze(1).expand(-1, sorted_indices.size(1))
            # sequence_indices = sorted_indices


            # # Use advanced indexing to retrieve the elements
            # x = x[batch_indices, sequence_indices]
            # print('xshape', x.shape, flush=True)
            x = x[:, sorted_indices, :]

            cfg['temp_input_ids'] = cfg['temp_input_ids'][:, sorted_indices]
            print('sorted_indices', sorted_indices, flush=True)
        decoded_string = cfg['tokenizer'].batch_decode(cfg['temp_input_ids'], skip_special_tokens=False)

        print(f'decoded_string_{probe_type}', decoded_string)
        return x, residual, sorted_indices
    else:
        # torch.set_printoptions(precision=2, sci_mode=False)
        if residual is not None:
            print('residual_ranking', flush=True)
            print(torch.isnan(residual).any(), torch.isinf(residual).any())

            if torch.isnan(residual).any():
                print('nan residual', flush=True)
                print(residual, flush=True)
                raise ValueError('nan residual')
            if 'bsz' in probe_type:
                l2_norms = torch.linalg.vector_norm(residual, ord=2, dim=(1, 2))
                # if values are the same, add little randomness to avoid the same probe
                # torch.float16 has the rounding issue
                noise = torch.randn(l2_norms.size(), device=l2_norms.device, dtype=l2_norms.dtype) * 0.5
                print('noise', noise, flush=True)
                l2_norms = l2_norms + noise
                print('l2_norms after noise', l2_norms, flush=True)
                print('l2_norms bsz', l2_norms, flush=True)
            elif 'seq' in probe_type:
                l2_norms = torch.linalg.vector_norm(residual, ord=2, dim=(0, 2))
                print('l2_norms seq', l2_norms, flush=True)
        else:
            if 'bsz' in probe_type:
                l2_norms = torch.linalg.vector_norm(x, ord=2, dim=(1, 2))
            elif 'seq' in probe_type:
                l2_norms = torch.linalg.vector_norm(x, ord=2, dim=(0, 2))

        

    # if 'seq' in probe_type:
    #     x_l2_norms = torch.linalg.vector_norm(x, ord=2, dim=2)
    #     l2_norms = torch.linalg.vector_norm(residual, ord=2, dim=2)
    # first_30_tokens = l2_norms[:, :30]
    # print('first_30_tokens l2_norms', first_30_tokens, flush=True)
    # first_30_x_tokens = x_l2_norms[:, :30]
    # print('first_30_tokens x_l2_norms', first_30_x_tokens, flush=True)


    # all_sorted_values, all_sorted_indices = torch.sort(l2_norms, descending=True)
    # all_sorted_values = all_sorted_values.to(torch.float32)
    # print('values', all_sorted_values, all_sorted_indices, flush=True)

    # sum_values = all_sorted_values.sum()
    # probabilities = (all_sorted_values / sum_values)
    # print('probabilities', probabilities, flush=True)
    # selected_indices = torch.multinomial(probabilities, probe_num, replacement=False)
    # all_sorted_values = all_sorted_values.to(torch.float16)
    # values = all_sorted_values[selected_indices]
    # indices = all_sorted_indices[selected_indices]
        print('x.size(1) - int(probe_num//2)', x.size(1) - int(probe_num//2))
        if 'seq' in probe_type:
            values, indices = torch.topk(l2_norms[:x.size(1) - int(probe_num//2)], int(probe_num//2))
            

            sorted_indices = indices.sort()[0]
            if sorted_indices[0] != 0:
                sorted_indices[0] = 0
            print(probe_type, sorted_indices, sorted_indices.shape, values)

            last_indices = torch.arange(x.size(1) - int(probe_num//2), x.size(1)).to(x.device)
            sorted_indices = torch.cat((sorted_indices, last_indices)).to(x.device)
            print('sorted_indices', sorted_indices, flush=True)
        else:
            values, indices = torch.topk(l2_norms, probe_num)
            print(probe_type, indices, values)

            sorted_indices = indices.sort()[0]
            print('sorted_indices', sorted_indices, flush=True)
    # sorted_values = values.sort()[0]
    # if 'seq' in probe_type:

    #     first_indices = torch.arange(1)

    #     # Calculate the start index for the last elements
    #     # start_index_from_end = max(4, last_dim_length - num_elements_from_end)  # Ensure there's no overlap with the first 4 indices

    #     # Generate indices for the last elements
    #     # probe_num = 1
    #     last_indices = torch.arange(x.shape[1] - probe_num, x.shape[1])

    #     # last_indices = torch.arange(30)

    #     # Combine the two sets of indices
    #     sorted_indices = torch.cat((first_indices, last_indices)).to(x.device)

    #     # sorted_indices = last_indices.to(x.device)
    #     # sorted_indices = first_indices.to(x.device)
    #     print('sorted_indices', sorted_indices, flush=True)
        cfg['temp_input_ids'] = cfg['temp_input_ids'].to(x.device)
        # print('cfg[temp_input_ids]', cfg['temp_input_ids'], x, flush=True)
        if 'bsz' in probe_type:
            cfg['temp_input_ids'] = cfg['temp_input_ids'][sorted_indices, :]
            # sorted_indices = torch.arange(probe_num)
            # sorted_indices = torch.randperm(x.size(0))[:probe_num]
            print('bsz_sorted_indices', sorted_indices, flush=True)
            x = x[sorted_indices, :, :]
        elif 'seq' in probe_type:      
            cfg['temp_input_ids'] = cfg['temp_input_ids'][:, sorted_indices]
            x = x[:, sorted_indices, :]

        if residual is not None:
            if 'bsz' in probe_type:
                residual = residual[sorted_indices, :, :]
            elif 'seq' in probe_type:
                residual = residual[:, sorted_indices, :]
        
        # print("cfg['temp_input_ids']", cfg['temp_input_ids'])
        decoded_string = cfg['tokenizer'].batch_decode(cfg['temp_input_ids'], skip_special_tokens=False)

        print(f'decoded_string_{probe_type}', decoded_string)
    return x, residual, sorted_indices

def generate_probe(x, probe_ratio_list, residual=None):
    # seq rank needs selected_indices to combine with the global metric
    bsz_selected_indices = None
    seq_selected_indices = None
    pad_tokens = cfg['pad_tokens']
    cfg['temp_input_ids'] = cfg['input_ids']
    print("cfg['temp_input_ids'] shape", cfg['temp_input_ids'].shape, cfg['input_ids'].shape, flush=True)
    print('x shape', x.shape, flush=True)

    # if 'mask' in cfg['prune_method']:
    #     pass
    # else:
        # if pad_tokens is not None:
        #     if residual is not None:
        #         residual[pad_tokens] = 0
        #         print('residual_generate_probe', id(residual), flush=True)
        #     else:
        #         x[pad_tokens] = 0

    for i in range(len(cfg['probe_generation_type'])):
        probe_type = cfg['probe_generation_type'][i]
        probe_ratio = probe_ratio_list[i]

        if 'bsz' in probe_type:
            probe_num = math.ceil(x.size(0) * probe_ratio)
        elif 'seq' in probe_type:
            probe_num = max(math.ceil(x.size(1) * probe_ratio), 2)
            if probe_num % 2 != 0:
                probe_num += 1
            probe_num = min(probe_num, x.size(1))


        if 'rank' in probe_type:
            x, residual, selected_indices = rank_process(x, probe_num, probe_type, residual)
            if 'bsz' in probe_type:
                bsz_selected_indices = selected_indices
            elif 'seq' in probe_type:
                seq_selected_indices = selected_indices
        else:
            raise NotImplementedError
    return x, bsz_selected_indices, seq_selected_indices

# torch.set_printoptions(threshold=1000)
# def rank_process(x, probe_num, probe_type, residual):
    
#     if residual is not None:
#         print('residual_ranking', flush=True)
#         print(torch.isnan(residual).any(), torch.isinf(residual).any())

#         if torch.isnan(residual).any():
#             print('nan residual', flush=True)
#             print(residual, flush=True)
#             raise ValueError('nan residual')
#         if 'bsz' in probe_type:
#             l2_norms = torch.linalg.vector_norm(residual, ord=2, dim=(1, 2))
#         elif 'seq' in probe_type:
#             l2_norms = torch.linalg.vector_norm(residual, ord=2, dim=(0, 2))
#     else:
#         if 'bsz' in probe_type:
#             l2_norms = torch.linalg.vector_norm(x, ord=2, dim=(1, 2))
#         elif 'seq' in probe_type:
#             l2_norms = torch.linalg.vector_norm(x, ord=2, dim=(0, 2))

#     print('l2_norms', l2_norms, flush=True)
#     # all_sorted_values, all_sorted_indices = torch.sort(l2_norms, descending=True)
#     # all_sorted_values = all_sorted_values.to(torch.float32)
#     # print('values', all_sorted_values, all_sorted_indices, flush=True)

#     # sum_values = all_sorted_values.sum()
#     # probabilities = (all_sorted_values / sum_values)
#     # print('probabilities', probabilities, flush=True)
#     # selected_indices = torch.multinomial(probabilities, probe_num, replacement=False)
#     # all_sorted_values = all_sorted_values.to(torch.float16)
#     # values = all_sorted_values[selected_indices]
#     # indices = all_sorted_indices[selected_indices]

#     values, indices = torch.topk(l2_norms, probe_num)
#     print(probe_type, indices, values)

#     sorted_indices = indices.sort()[0]
#     # sorted_values = values.sort()[0]
#     # if 'seq' in probe_type:

#     #     first_indices = torch.arange(1)

#     #     # Calculate the start index for the last elements
#     #     # start_index_from_end = max(4, last_dim_length - num_elements_from_end)  # Ensure there's no overlap with the first 4 indices

#     #     # Generate indices for the last elements
#     #     last_indices = torch.arange(int(x.shape[1]//2) - probe_num, int(x.shape[1]//2))

#     #     # Combine the two sets of indices
#     #     sorted_indices = torch.cat((first_indices, last_indices)).to(x.device)

#     #     # sorted_indices = last_indices.to(x.device)
#     #     print('sorted_indices', sorted_indices, flush=True)

#     if 'bsz' in probe_type:
#         return x[sorted_indices, :, :], sorted_indices
#     elif 'seq' in probe_type:        
#         sorted_indices = sorted_indices.unsqueeze(0).repeat(x.shape[0], 1)
#         # print('sorted_indices', sorted_indices, flush=True)
#         # sorted_indices[:, 0] = cfg['first_one_indices']
#         # print('sorted_indices after', sorted_indices, flush=True)

#         # sorted_indices = sorted_indices.unsqueeze(-1)
#         # new_x = torch.gather(x, 1, sorted_indices.expand(-1, -1, x.shape[2]))

#         # print('sorted_indices after update:', sorted_indices.squeeze(-1))
#         # print('new_x shape:', new_x.shape)

#         # # Returning new_x and sorted_indices for further use
#         # return new_x, sorted_indices.squeeze(-1)
#         for i in range(x.shape[0]):
#             sorted_indices[i][0] = cfg['first_one_indices'][i][0]
#             # sorted_indices[i].sort_()
#             sorted_indices[i], _ = sorted_indices[i].sort() 

#         # Correct indexing
#         # You need to ensure each batch item uses its corresponding sorted_indices for advanced indexing
#         new_x = torch.stack([x[i, sorted_indices[i], :] for i in range(x.shape[0])])

#         # print('sorted_indices', sorted_indices)
#         # print('new_x shape:', new_x)

#         return new_x, sorted_indices

#         return x[:, sorted_indices, :], sorted_indices


# def generate_probe(x, probe_ratio_list, residual=None):
#     # seq rank needs selected_indices to combine with the global metric
#     bsz_selected_indices = None
#     seq_selected_indices = None
#     pad_tokens = cfg['pad_tokens']

#     if 'mask' in cfg['prune_method']:
#         pass
#     else:
#         if pad_tokens is not None:
#             if residual is not None:
#                 residual[pad_tokens] = 0
#             else:
#                 x[pad_tokens] = 0

#     for i in range(len(cfg['probe_generation_type'])):
#         probe_type = cfg['probe_generation_type'][i]
#         probe_ratio = probe_ratio_list[i]

#         if 'bsz' in probe_type:
#             probe_num = math.ceil(x.size(0) * probe_ratio)
#         elif 'seq' in probe_type:
#             probe_num = max(math.ceil(x.size(1) * probe_ratio), 1)

#         if 'rank' in probe_type:
#             x, selected_indices = rank_process(x, probe_num, probe_type, residual)
#             if 'bsz' in probe_type:
#                 bsz_selected_indices = selected_indices
#             elif 'seq' in probe_type:
#                 seq_selected_indices = selected_indices
#         else:
#             raise NotImplementedError
#     # print('probeshape', x.shape, bsz_selected_indices, flush=True)
#     seq_selected_indices = seq_selected_indices[bsz_selected_indices].squeeze(0)
#     # print('seq_selected_indices', seq_selected_indices, seq_selected_indices.shape, flush=True)
#     return x, bsz_selected_indices, seq_selected_indices
    

def check_nan_inf(x):
    if torch.isnan(x).any():
        print('nan', flush=True)
        print(x, torch.max(x), torch.min(x), flush=True)
        raise ValueError('nan')
    if torch.isinf(x).any():
        print('inf', flush=True)
        print(x, torch.max(x), torch.min(x), flush=True)
        raise ValueError('inf')



def cal_res_hidden_state_diff(hidden_states, residual):
    # cfg['resinfo_ratio'] = 0.8
    print('resinfo_ratio', cfg['resinfo_ratio'], flush=True)
    # torch.set_printoptions(threshold=float('inf')) 
    flattened_hidden_states = hidden_states.flatten()
    print('flattened_hidden_states', flattened_hidden_states, flush=True)
    num_elements_to_select = max(1, int(cfg['resinfo_ratio']* flattened_hidden_states.numel()))  # Top 10% of elements
    # Select the top 10% elements based on their absolute value
    abs_flattened_hidden_states = -flattened_hidden_states.abs()
    values, indices = torch.topk(abs_flattened_hidden_states, num_elements_to_select)

    ## Retrieve the actual values from the original tensor using these indices
    selected_hidden_values = flattened_hidden_states[indices]
    # print('selected_hidden_values', selected_hidden_values, flush=True)
    flattened_residual = residual.flatten()
    selected_residual = flattened_residual[indices]


    # calculate sign match percentage
    sign_matches = torch.sign(selected_hidden_values) == torch.sign(selected_residual)
    sign_match_percentage = torch.sum(sign_matches).item() / num_elements_to_select * 100

    # calculate l1 difference percentage
    # to float32 avoid overflow
    selected_hidden_values = selected_hidden_values.to(torch.float32)
    selected_residual = selected_residual.to(torch.float32)
    l1_norm = selected_hidden_values.abs().sum()
    l2_norm_hidden_values = torch.linalg.vector_norm(selected_hidden_values, ord=2)

    l2_norm_residual = torch.linalg.vector_norm(selected_residual, ord=2)
    # l1_diff_norm = (selected_hidden_values - selected_residual).abs().sum()
    # l1_diff_percentage = (l2_norm_residual.item() - l2_norm_hidden_values.item()) / l2_norm_hidden_values.item() * 100

#     magnitude_ratio = np.linalg.norm(A) / np.linalg.norm(B) if np.linalg.norm(B) > np.linalg.norm(A) else np.linalg.norm(B) / np.linalg.norm(A)
# print("Magnitude Ratio:", magnitude_ratio)

    if l2_norm_hidden_values.item() > l2_norm_residual.item():
        l2_magnitude_ratio = l2_norm_residual.item() / l2_norm_hidden_values.item()
    else:
        l2_magnitude_ratio = l2_norm_hidden_values.item() / l2_norm_residual.item()
    
    # sort_residual, sort_residual_indices = torch.sort(selected_residual)
    # sort_hidden, sort_hidden_indices = torch.sort(selected_hidden_values)
    
    # print('sort_residual', sort_residual, sort_residual_indices, flush=True)
    # print('sort_hidden', sort_hidden, sort_hidden_indices, flush=True)
    # for val in sort_residual[:100]:
    #     print(val, 'sort_residual')
    # for val in sort_residual_indices[:100]:
    #     print(val, 'sort_residual_indices')

    # for val in sort_hidden[:100]:
    #     print(val, 'sort_hidden')
    # for val in sort_hidden_indices[:100]:
    #     print(val, 'sort_hidden_indices')

    # for val in sort_residual[-100:]:
    #     print(val, 'sort_residual')
    # for val in sort_residual_indices[-100:]:
    #     print(val, 'sort_residual_indices')
    # for val in sort_hidden[-100:]:
    #     print(val, 'sort_hidden')
    # for val in sort_hidden_indices[-100:]:
    #     print(val, 'sort_hidden_indices')
    # calculate cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(
        selected_hidden_values,  # Ensure the data type is float for cosine similarity computation
        selected_residual,
        dim=0  # Compute the cosine similarity across the dimension 0 (element-wise for vectors)
    ).item()


    return sign_match_percentage, l2_magnitude_ratio, cosine_similarity













# def rank_process(x, probe_num, probe_size, probe_type, residual):
#     if residual is not None:
#         if 'bsz' in probe_type:
#             l2_norms = torch.linalg.vector_norm(residual, ord=2, dim=(1, 2))
#         elif 'seq' in probe_type:
#             l2_norms = torch.linalg.vector_norm(residual, ord=2, dim=(0, 2))
#         elif 'hd' in probe_type:
#             l2_norms = torch.linalg.vector_norm(residual, ord=2, dim=(0, 1))
#     else:
#         if 'bsz' in probe_type:
#             l2_norms = torch.linalg.vector_norm(x, ord=2, dim=(1, 2))
#         elif 'seq' in probe_type:
#             l2_norms = torch.linalg.vector_norm(x, ord=2, dim=(0, 2))
#         elif 'hd' in probe_type:
#             l2_norms = torch.linalg.vector_norm(x, ord=2, dim=(0, 1))

#     values, indices = torch.topk(l2_norms, probe_num)
#     sorted_indices = indices.sort()[0]

#     if 'bsz' in probe_type:
#         return x[sorted_indices, :, :], sorted_indices
#     elif 'seq' in probe_type:        
#         return x[:, sorted_indices, :], sorted_indices
#     elif 'hd' in probe_type:
#         return x[:, :, sorted_indices], sorted_indices


# def mean_process(x, probe_num, probe_size, probe_type):
#     if 'bsz' in probe_type:
#         probe = torch.mean(x.view(probe_num, probe_size, x.size(-2), x.size(-1)), dim=1)
#     return probe


# def absnml_process(x, probe_num, probe_size, probe_type):
#     abs_x = torch.abs(x).clamp_min_(cfg['data_type_min_positive'])
#     if 'bsz' in probe_type:
#         abs_view = abs_x.view(probe_num, probe_size, x.size(-2), x.size(-1))
#         probe = (x.view(probe_num, probe_size, x.size(-2), x.size(-1)) * (abs_view / abs_view.sum(dim=1, keepdim=True))).sum(dim=1)
#     return probe


# def cut_extra_dim(x, probe_type, probe_num, probe_size, residual):
#     if 'bsz' in probe_type:
#         if x.size(0) % probe_num != 0:
#             x = x[:probe_num * probe_size, :, :]
#             if residual is not None:
#                 residual = residual[:probe_num * probe_size, :, :]
#     elif 'seq' in probe_type:
#         if x.size(1) % probe_num != 0:
#             x = x[:, :probe_num * probe_size, :]
#             if residual is not None:
#                 residual = residual[:, :probe_num * probe_size, :]
#     elif 'hd' in probe_type:
#         if x.size(2) % probe_num != 0:
#             x = x[:, :, :probe_num * probe_size]
#             if residual is not None:
#                 residual = residual[:, :, :probe_num * probe_size]
#     return x, residual

# def generate_probe(x, probe_ratio_list, residual=None):
#     # seq rank needs selected_indices to combine with the global metric
#     bsz_selected_indices = None
#     seq_selected_indices = None
#     pad_tokens = cfg['pad_tokens']

#     if pad_tokens is not None:
#         if residual is not None:
#             residual[pad_tokens] = 0
#         else:
#             x[pad_tokens] = 0

#     for i in range(len(cfg['probe_generation_type'])):
#         probe_type = cfg['probe_generation_type'][i]
#         probe_ratio = probe_ratio_list[i]

#         if 'bsz' in probe_type:
#             probe_num = int(x.size(0) * probe_ratio)
#             probe_size = x.size(0) // probe_num
#         elif 'seq' in probe_type:
#             probe_num = int(x.size(1) * probe_ratio)
#             probe_size = x.size(1) // probe_num
#         elif 'hd' in probe_type:
#             probe_num = int(x.size(2) * probe_ratio)
#             probe_size = x.size(2) // probe_num
        
#         # might need to cut extra dim to use view()
#         if 'mean' in probe_type or 'absnml' in probe_type:
#             x, residual = cut_extra_dim(x, probe_type, probe_num, probe_size, residual)

#         if 'rank' in probe_type:
#             x, selected_indices = rank_process(x, probe_num, probe_size, probe_type, residual)
#             if 'bsz' in probe_type:
#                 bsz_selected_indices = selected_indices
#             elif 'seq' in probe_type:
#                 seq_selected_indices = selected_indices
#         elif 'mean' in probe_type:
#             x = mean_process(x, probe_num, probe_size, probe_type)
#         elif 'absnml' in probe_type:
#             x = absnml_process(x, probe_num, probe_size, probe_type)
#     return x, bsz_selected_indices, seq_selected_indices
    








# def nml_process(x, probe_num, probe_size):
#     # avoid nan proportion
#     abs_x = torch.clamp(torch.abs(x), min=1e-6)
#     sum_across_bsz = abs_x.view(probe_num, probe_size, x.size(-2), x.size(-1)).sum(dim=1, keepdim=True)
#     proportion = abs_x.view(probe_num, probe_size, x.size(-2), x.size(-1)) / sum_across_bsz
#     comp_across_bsz = (x.view(probe_num, probe_size, x.size(-2), x.size(-1)) * proportion).sum(dim=1)
#     return comp_across_bsz





# def max_process(x, probe_num, probe_size):
#     # Apply absolute value to x
#     abs_x = torch.abs(x)
#     # Adjust the view to organize the data by probe_num and probe_size
#     reorganized_x = x.view(probe_num, probe_size, x.size(-2), x.size(-1))
#     reorganized_abs_x = abs_x.view(probe_num, probe_size, x.size(-2), x.size(-1))
#     # Use torch.max to get the indices of maximum value across the probe_size dimension
#     _, indices = reorganized_abs_x.max(dim=1, keepdim=True)
#     # Use these indices to gather the original values from reorganized_x
#     max_across_bsz = torch.gather(reorganized_x, 1, indices).squeeze(1)
#     # print('max_across_bsz', max_across_bsz.shape, flush=True)
#     return max_across_bsz




















    #  if self.input_norm_gate_weight is None:
    #             self.input_norm_gate_weight = torch.linalg.vector_norm(self.gate_proj.weight.data, ord=2, dim=0).reshape(1, 1, -1)
    #         if self.input_norm_up_weight is None:
    #             self.input_norm_up_weight = torch.linalg.vector_norm(self.up_proj.weight.data, ord=2, dim=0).reshape(1, 1, -1)

    #         # def cal_sign_agreement_metrix(x_flattened):
    #         #     strength = torch.linalg.vector_norm(x_flattened, ord=2, dim=0)
    #         #     # Calculate the number of positions to select (top 10%)
    #         #     top_k = max(int(0.03 * strength.numel()), 1)  # Ensure at least one position is selected
    #         #     # print('top_k', top_k, strength.numel(), strength.shape, flush=True)
    #         #     # Use torch.topk to find the top k positions. 
    #         #     # torch.topk returns values and their corresponding indices.
    #         #     top_values, top_indices = torch.topk(strength, k=top_k)

    #         #     top_positions_flat = x_flattened[:, top_indices]  # [bsz, top_k]
    #         #     print('top_positions_flat', top_positions_flat, flush=True)
    #         #     signs_top_positions_flat = torch.sign(top_positions_flat)
    #         #     # print('signs_top_positions_flat', signs_top_positions_flat, flush=True)
    #         #     # sign_similarity = signs_top_positions_flat * signs_top_positions_flat.transpose(0, 1)
    #         #     # Expand dimensions for broadcasting
    #         #     expanded_signs = signs_top_positions_flat.unsqueeze(1)  # Shape: [bsz, 1, top_k]
    #         #     # Repeat signs for comparison across all pairs
    #         #     repeated_signs = signs_top_positions_flat.unsqueeze(0)  # Shape: [1, bsz, top_k]

    #         #     # Element-wise multiplication to check sign agreement (-1 * -1 = 1, 1 * 1 = 1, else = -1 or 0)
    #         #     sign_agreement = expanded_signs * repeated_signs  # Shape: [bsz, bsz, top_k]

    #         #     # Sum over the top_k dimension to count the number of agreements per pair
    #         #     sign_agreement_matrix = sign_agreement.sum(dim=-1)  # Shape: [bsz, bsz]
    #         #     # print('sign_agreement_matrix v1', sign_agreement_matrix, flush=True)
    #         #     sign_agreement_matrix = sign_agreement_matrix / top_k
    #         #     torch.set_printoptions(threshold=5000)  # Adjust the number as needed
    #         #     print('sign_agreement_matrix', sign_agreement_matrix, flush=True)
    #         # #     print('top_positions_flat', top_positions_flat, flush=True)
    #         # #     # Normalize
    #         # #     norm_signs_top_positions_flat = signs_top_positions_flat / (torch.linalg.vector_norm(signs_top_positions_flat, ord=2, dim=-1, keepdim=True) + 1e-9)
    #         # #    # Assuming norm_top_positions_flat is [bsz, top_k]
    #         # #     similarity_matrix = torch.matmul(norm_signs_top_positions_flat, norm_signs_top_positions_flat.transpose(0, 1))
            
    #         # x_temp_gate = x * self.input_norm_gate_weight
    #         # x_temp_up = x * self.input_norm_up_weight
    #         # print('\nx_temp_gate')
    #         # cal_sign_agreement_metrix(x_temp_gate.view(x_temp_gate.size(0), -1))
    #         # print('\nx_temp_up')
    #         # cal_sign_agreement_metrix(x_temp_up.view(x_temp_up.size(0), -1))
    #         # x_flattened = x.view(x.size(0), -1) 
    #         def cal_dot_product_matrix(x_flattened):
    #             strength = torch.linalg.vector_norm(x_flattened, ord=2, dim=0)
    #             # Calculate the number of positions to select (top 3% here as per your code)
    #             top_k = max(int(0.03 * strength.numel()), 1)  # Ensure at least one position is selected
    #             top_values, top_indices = torch.topk(strength, k=top_k)

    #             top_positions_flat = x_flattened[:, top_indices]  # [bsz, top_k]
                
    #             # Calculate dot product matrix
    #             # Normalize the vectors to only measure directionality
    #             # norm_top_positions_flat = top_positions_flat / (torch.linalg.vector_norm(top_positions_flat, ord=2, dim=-1, keepdim=True) + 1e-9)
                
    #             # Dot product similarity (using matrix multiplication for efficiency)
    #             # Here, we're effectively doing dot product because the vectors are normalized
    #             dot_product_matrix = torch.matmul(top_positions_flat, top_positions_flat.transpose(0, 1))  # Shape: [bsz, bsz]
                
    #             # Optionally, normalize the dot product matrix to scale the values between -1 and 1
    #             # This step may not be necessary since we're already working with normalized vectors
    #             # dot_product_matrix = dot_product_matrix / top_k  # Normalize if needed
                
    #             torch.set_printoptions(threshold=5000)  # Adjust the number as needed
    #             print('dot_product_matrix', dot_product_matrix, flush=True)

    #         # Example of how to call the function
    #         # Assuming x is your input tensor
    #         x_temp_gate = x * self.input_norm_gate_weight
    #         x_temp_up = x * self.input_norm_up_weight
    #         # Flatten x as needed and pass to the function
    #         cal_dot_product_matrix(x_temp_gate.view(x_temp_gate.size(0), -1))
    #         cal_dot_product_matrix(x_temp_up.view(x_temp_up.size(0), -1))

    #         end_time = time.time()
    #         print('similarity_duration', self.layer_order, end_time - start_time, flush=True)