import math
import torch
from config import cfg


def get_next_layer(layers, idx):
    def get_layer_device(layer):
        return next(layer.parameters()).device

    if idx + 1 < len(layers) and get_layer_device(layers[idx + 1]) == get_layer_device(layers[idx]):
        next_layer = layers[idx + 1]
    elif idx + 1 < len(layers) and get_layer_device(layers[idx + 1]) != get_layer_device(layers[idx]):
        next_layer = None
    elif idx + 1 == len(layers):
        next_layer = None
    
    return next_layer


def rank_process(norm_across_feature, probe_num, probe_type):
    if 'bsz' in probe_type:
        l2_norms = torch.linalg.vector_norm(norm_across_feature, ord=2, dim=1)
    elif 'seq' in probe_type:
        l2_norms = torch.linalg.vector_norm(norm_across_feature, ord=2, dim=0)
    
    values, indices = torch.topk(l2_norms, probe_num)
    sorted_indices = indices.sort()[0]
    if 'seq' in probe_type and probe_num <= 10:
        # keep the first token in the extreme case
        sorted_indices[0] = 0
        
    return sorted_indices

def get_norm_across_feature(x, residual=None):
    if residual is not None:
        return torch.linalg.vector_norm(residual, ord=2, dim=2)
    return torch.linalg.vector_norm(x, ord=2, dim=2)

def generate_probe(x, probe_ratio_list, residual=None):
    bsz_selected_indices = None
    seq_selected_indices = None

    norm_across_feature = get_norm_across_feature(x, residual)
    for i in range(len(cfg['probe_generation_type'])):
        probe_type = cfg['probe_generation_type'][i]
        probe_ratio = probe_ratio_list[i]

        if 'bsz' in probe_type:
            probe_num = math.ceil(x.size(0) * probe_ratio)
        elif 'seq' in probe_type:
            probe_num = max(math.ceil(x.size(1) * probe_ratio), 2)

        if 'rank' in probe_type:
            if 'bsz' in probe_type:
                bsz_selected_indices = rank_process(norm_across_feature, probe_num, probe_type)
                norm_across_feature = norm_across_feature[bsz_selected_indices, :]
            elif 'seq' in probe_type:
                seq_selected_indices = rank_process(norm_across_feature, probe_num, probe_type)
                norm_across_feature = norm_across_feature[:, seq_selected_indices]
        else:
            raise NotImplementedError
 
    if bsz_selected_indices is not None and seq_selected_indices is not None:
        ii, jj = torch.meshgrid(bsz_selected_indices, seq_selected_indices, indexing='ij')
        probe = x[ii, jj, :]
    elif bsz_selected_indices is not None:
        probe = x[bsz_selected_indices, :, :]
    elif seq_selected_indices is not None:
        probe = x[:, seq_selected_indices, :]
    else:
        raise ValueError('No selected indices')
    return probe, bsz_selected_indices, seq_selected_indices


def check_nan_inf(x):
    if torch.isnan(x).any():
        print('nan', flush=True)
        print(x, torch.max(x), torch.min(x), flush=True)
        raise ValueError('nan')
    if torch.isinf(x).any():
        print('inf', flush=True)
        print(x, torch.max(x), torch.min(x), flush=True)
        raise ValueError('inf')