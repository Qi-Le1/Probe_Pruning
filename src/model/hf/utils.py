import torch


def nml_process(x, probe_num, probe_size):
    # avoid nan proportion
    abs_x = torch.clamp(torch.abs(x), min=1e-6)
    sum_across_bsz = abs_x.view(probe_num, probe_size, x.size(-2), x.size(-1)).sum(dim=1, keepdim=True)
    proportion = abs_x.view(probe_num, probe_size, x.size(-2), x.size(-1)) / sum_across_bsz
    comp_across_bsz = (x.view(probe_num, probe_size, x.size(-2), x.size(-1)) * proportion).sum(dim=1)
    return comp_across_bsz

def max_process(x, probe_num, probe_size):
    # Apply absolute value to x
    abs_x = torch.abs(x)
    # Adjust the view to organize the data by probe_num and probe_size
    reorganized_x = x.view(probe_num, probe_size, x.size(-2), x.size(-1))
    reorganized_abs_x = abs_x.view(probe_num, probe_size, x.size(-2), x.size(-1))
    # Use torch.max to get the indices of maximum value across the probe_size dimension
    _, indices = reorganized_abs_x.max(dim=1, keepdim=True)
    # Use these indices to gather the original values from reorganized_x
    max_across_bsz = torch.gather(reorganized_x, 1, indices).squeeze(1)
    # print('max_across_bsz', max_across_bsz.shape, flush=True)
    return max_across_bsz