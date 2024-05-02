import math
import torch
from config import cfg


def rank_process(x, probe_num, probe_type, residual):
    if residual is not None:
        if 'bsz' in probe_type:
            l2_norms = torch.linalg.vector_norm(residual, ord=2, dim=(1, 2))
        elif 'seq' in probe_type:
            l2_norms = torch.linalg.vector_norm(residual, ord=2, dim=(0, 2))
    else:
        if 'bsz' in probe_type:
            l2_norms = torch.linalg.vector_norm(x, ord=2, dim=(1, 2))
        elif 'seq' in probe_type:
            l2_norms = torch.linalg.vector_norm(x, ord=2, dim=(0, 2))

    values, indices = torch.topk(l2_norms, probe_num)
    sorted_indices = indices.sort()[0]

    if 'bsz' in probe_type:
        return x[sorted_indices, :, :], sorted_indices
    elif 'seq' in probe_type:        
        return x[:, sorted_indices, :], sorted_indices


def generate_probe(x, probe_ratio_list, residual=None):
    # seq rank needs selected_indices to combine with the global metric
    bsz_selected_indices = None
    seq_selected_indices = None
    pad_tokens = cfg['pad_tokens']

    if pad_tokens is not None:
        if residual is not None:
            residual[pad_tokens] = 0
        else:
            x[pad_tokens] = 0

    for i in range(len(cfg['probe_generation_type'])):
        probe_type = cfg['probe_generation_type'][i]
        probe_ratio = probe_ratio_list[i]

        if 'bsz' in probe_type:
            probe_num = math.ceil(x.size(0) * probe_ratio)
        elif 'seq' in probe_type:
            probe_num = math.ceil(x.size(1) * probe_ratio)

        if 'rank' in probe_type:
            x, selected_indices = rank_process(x, probe_num, probe_type, residual)
            if 'bsz' in probe_type:
                bsz_selected_indices = selected_indices
            elif 'seq' in probe_type:
                seq_selected_indices = selected_indices
        else:
            raise NotImplementedError
    return x, bsz_selected_indices, seq_selected_indices
    

















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



def cal_res_hidden_state_diff(hidden_states, residual):
    flattened_hidden_states = hidden_states.flatten()
    num_elements_to_select = max(1, int(cfg['resinfo_ratio']* flattened_hidden_states.numel()))  # Top 10% of elements
    # Select the top 10% elements based on their absolute value
    abs_flattened_hidden_states = flattened_hidden_states.abs()
    values, indices = torch.topk(abs_flattened_hidden_states, num_elements_to_select)

    ## Retrieve the actual values from the original tensor using these indices
    selected_hidden_values = flattened_hidden_states[indices]

    flattened_residual = residual.flatten()
    selected_residual = flattened_residual[indices]


    # calculate sign match percentage
    sign_matches = torch.sign(selected_hidden_values) == torch.sign(selected_residual)
    sign_match_percentage = torch.sum(sign_matches).item() / num_elements_to_select * 100

    # calculate l1 difference percentage
    selected_hidden_values = selected_hidden_values.to(torch.float32)
    selected_residual = selected_residual.to(torch.float32)
    l1_norm = selected_hidden_values.abs().sum()
    l1_diff_norm = (selected_hidden_values - selected_residual).abs().sum()
    l1_diff_percentage = l1_diff_norm.item() / l1_norm.item() * 100

    # calculate cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(
        selected_hidden_values,  # Ensure the data type is float for cosine similarity computation
        selected_residual,
        dim=0  # Compute the cosine similarity across the dimension 0 (element-wise for vectors)
    ).item()


    return sign_match_percentage, l1_diff_percentage, cosine_similarity

















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