import torch 
import torch.nn as nn 
from .layerwrapper import WrappedGPT, BiasGPT
# from .data import get_loaders 
import math
import re
import numpy as np
from config import cfg
from tqdm import tqdm
from .ewi import Linear
from module import to_device, TRANSFORMERS_MODELS_TO_EWI_TARGET_MODULES_MAPPING, to_device
# create a dictionary to map the method name to the function
"""
    'WIFN': Weighted Input Feature Norm
    'IFV': Input Feature Variance
    'WIFV': Weighted Input Feature Variance
    
"""

metrics = {
    # 'IFN': lambda wrapped_layers, subset, name: torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1))).squeeze(0),
    'wanda': lambda wrapped_layers, subset, name: (torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1))) * torch.abs(subset[name].weight.data)).mean(axis=0),
    # out product
    # 'O1WIFN': lambda wrapped_layers, subset, name: torch.sum((torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1, -1))).reshape(-1, 1) * torch.linalg.vector_norm(subset[name].weight.data, ord=1, dim=1).reshape(1, -1)), dim=1),
    # 'O2WIFN': lambda wrapped_layers, subset, name: torch.sum((torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1, -1))).reshape(-1, 1) * torch.linalg.vector_norm(subset[name].weight.data, ord=2, dim=1).reshape(1, -1)), dim=1),
    # 'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'flap': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
}


def if_add_bias():
    if 'bias' in cfg['prune_name']:
        return True
    return False

def if_standardize():
    if 'std' in cfg['prune_name']:
        return True
    return False

def if_normalize():
    if 'nml' in cfg['prune_name']:
        return True
    return False

def if_global_prune():
    if 'global' in cfg['prune_name']:
        return True
    return False

class MySettings:
    def __init__(self):
        self.add_bias = if_add_bias()
        self.global_prune = if_global_prune()
        self.standardize = if_standardize()
        self.normalize = if_normalize()


def metric_process(mysetting, x):
    if mysetting.standardize:
        print('metric process standardize')
        return (x - torch.mean(x, axis=-1, keepdim=True)) / torch.std(x, axis=-1, keepdim=True)
    elif mysetting.normalize:
        print('metric process normalize')
        return x / torch.sum(x, axis=-1, keepdim=True)
    else:
        print('no metric process')
        return x
        

def _check_target_module_exists(target_modules, key):
    if isinstance(target_modules, str):
        target_module_found = re.fullmatch(target_modules, key)
    else:
        target_module_found = any(key.endswith(target_key) for target_key in target_modules)
    return target_module_found

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def _get_target_modules(cfg):
    target_modules = TRANSFORMERS_MODELS_TO_EWI_TARGET_MODULES_MAPPING[cfg['model_type']]
    if 'cust_tgt_modules' in cfg and 'default' not in cfg['cust_tgt_modules']:
        target_modules = cfg['cust_tgt_modules']
    return target_modules

def find_layers(module, layers=[nn.Linear, Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    # print('name', name)
    # print('type(module)', type(module), module)
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def calibrate_model(model, tokenizer, dataloader, device):
    logger_info = {}
    if 'llama' in cfg['model_name']:
        if 'flap' in cfg['prune_name']: 
            prune_flap_llama(model, tokenizer, dataloader, logger_info, device)
        elif "wandasp" in cfg['prune_name']:
            prune_wanda_sp_llama(model, tokenizer, dataloader, logger_info, device)
        elif "magsp" in cfg['prune_name']:
            prune_magnitude_sp_llama(model, tokenizer, dataloader, logger_info, device)
        elif "pq" in cfg['prune_name']:
            prune_pq_llama(model, tokenizer, dataloader, logger_info, device)
        else:
            raise ValueError('Not valid prune_name')

    print("*"*30)
    whole_model_sparsity_ratio, pruned_module_sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check whole model: {whole_model_sparsity_ratio:.4f}")
    print(f"sparsity sanity check pruned module: {pruned_module_sparsity_ratio:.4f}")
    print(f"model parameter {sum(p.numel() for p in model.parameters()) / 1024 ** 3:.2f}B")
    print("*"*30)
    return logger_info

def check_sparsity(model):
    """
    Check the sparsity of the weights in different layers of the model.
    
    Args:
        model (nn.Module): The model to check.
        
    Returns:
        float: Ratio of the count of non-zero weights to total parameters in the model.
    """
    target_modules = _get_target_modules(cfg)
    print('target_modules', target_modules)
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    
    count = 0 
    total_params = 0

    attn_count = 0
    attn_params = 0

    mlp_count = 0
    mlp_params = 0

    pruned_module_count = 0
    pruned_module_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0

        attn_sub_count = 0
        attn_sub_params = 0

        mlp_sub_count = 0
        mlp_sub_params = 0
        for name in subset:
            print('sparsity check name', name)
            W = subset[name].weight.data
            sub_count += W.numel()
            count += W.numel()

            if 'self_attn' in name:
                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size

                attn_sub_count += W.numel()
                attn_sub_params += hidden_size * hidden_size

                attn_count += W.numel()
                attn_params += hidden_size * hidden_size
            else:
                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size

                mlp_sub_count += W.numel()
                mlp_sub_params += hidden_size * intermediate_size

                mlp_count += W.numel()
                mlp_params += hidden_size * intermediate_size
            if subset[name].bias is not None:
                count += subset[name].bias.data.numel()
                sub_count += subset[name].bias.data.numel()
            
            # key_list = [key for key, _ in layer.named_modules()]
            # # print('key_list', key_list)
            # for key in key_list:
            if _check_target_module_exists(target_modules, name):
                
                if 'self_attn' in name:
                    # hardcode for * 4: 4 attn layer in llama each module
                    pruned_module_count += W.numel() * 4
                    pruned_module_params += hidden_size * hidden_size * 4
                else:
                    pruned_module_count += W.numel() * 3
                    pruned_module_params += hidden_size * intermediate_size * 3

                if subset[name].bias is not None:
                    pruned_module_count += subset[name].bias.data.numel()
                    pruned_module_params += subset[name].bias.data.numel()

        print(f"layer {i} attn sparsity {float(attn_sub_count)/attn_sub_params:.6f}")
        print(f"layer {i} mlp sparsity {float(mlp_sub_count)/mlp_sub_params:.6f}")
        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        print(f'attn prune para {attn_params - attn_count}', f'mlp prune para {mlp_params - mlp_count}')

        print('attn para ratio: ', attn_sub_params / sub_params)
        print('mlp para ratio: ', mlp_sub_params / sub_params)
        print('-------\n')
        

    model.config.use_cache = use_cache 
    print(pruned_module_count, pruned_module_params)
    return float(count)/total_params, float(pruned_module_count)/pruned_module_params


def prepare_calibration_input(model, dataloader, device):
    """
    Prepare inputs for model calibration. 
    
    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded. 
        
    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    # print('fypemodel', type(model))
    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((cfg['nsamples'], cfg[cfg['model_name']]['max_length'], model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # print('inforward')
            # print('catcherinp', inp, cache['i'])
            # print("cache['i']", cache['i'])
            # print('kwargs', kwargs)
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    # print('layers[0]', layers[0])
    for i, batch in enumerate(dataloader):
        # print('batch', batch)
        try:
            batch_on_device = to_device(batch, device)
            model(batch_on_device['input_ids'])
        except ValueError as e:
            print('ValueError: ', e)
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    # print('catcherinps', inps, outs, attention_mask, position_ids )
    return inps, outs, attention_mask, position_ids 


def compress(idx, layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
    """
    Compress a model layer by masking or pruning based on the given masks.
    
    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.
        
    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    if unstr:  # Only mask, do not really prune
        # Attention Weight Masking
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(128)
            # Apply the mask to the query, key and value projection weights
            layer.self_attn.q_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            layer.self_attn.k_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            layer.self_attn.v_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            
            output_weight = layer.self_attn.o_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)
                
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                
                layer.self_attn.o_proj.bias.data = output_bias
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Masking
        if mlp_mask is not None:
            # Apply the mask to the up and gate projection weights
            layer.mlp.up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            layer.mlp.gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            
            output_weight = layer.mlp.down_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
                
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                layer.mlp.down_proj.bias.data = output_bias
            layer.mlp.down_proj.weight.data = output_weight
    
    else:
        # Real Pruning
        # Attention Weight Pruning
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(128)
            
            # Prune the query, key and value projection weights
            # We reduce the size of the weights based on the attention mask
            layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]]
            
            # Update output dimensions of q, k, v projections based on remaining heads
            layer.self_attn.q_proj.out_features = attn_mask.sum().item()
            layer.self_attn.k_proj.out_features = attn_mask.sum().item()
            layer.self_attn.v_proj.out_features = attn_mask.sum().item()
            
            output_weight = layer.self_attn.o_proj.weight.data
            
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)
                
            # Prune the output projection weight
            output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]
            # Update layer configurations for the new output shape after pruning
            # num_heads and num_key_value_heads currently have the same values
            layer.self_attn.num_heads = retain_heads
            layer.self_attn.num_key_value_heads = retain_heads
            
            # for attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            layer.self_attn.hidden_size = retain_heads * 128
            
            layer.self_attn.q_proj.is_pruned = True
            layer.self_attn.k_proj.is_pruned = True
            layer.self_attn.v_proj.is_pruned = True
            layer.self_attn.o_proj.is_pruned = True

            layer.self_attn.q_proj.key = f'{idx}.self_attn.q_proj'
            layer.self_attn.k_proj.key = f'{idx}.self_attn.k_proj'
            layer.self_attn.v_proj.key = f'{idx}.self_attn.v_proj'
            layer.self_attn.o_proj.key = f'{idx}.self_attn.o_proj'
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                layer.self_attn.o_proj.in_features = attn_mask.sum().item()
                # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                print('output_bias.shape', output_bias.shape)
                # if layer.self_attn.o_proj.bias is None:
                layer.self_attn.o_proj.bias = nn.Parameter(torch.zeros(output_bias.shape, device=device))
                layer.self_attn.o_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Pruning
        if mlp_mask is not None:
            # Prune the up and gate projection weights
            # print('before pruning layer.mlp.up_proj', layer.mlp.up_proj.weight.data.shape)
            # for i in range(layer.mlp.up_proj.weight.data.shape[0]):
            #     if i < 20:
            #         print('before pruning layer.mlp.up_proj', layer.mlp.up_proj.weight.data[i, :100])
            #     else:
            #         break
            # print('layer.mlp.up_proj.weight.data.shape', layer.mlp.up_proj.weight.data.shape, torch.where(mlp_mask)[0].shape)
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
            # print('after pruning layer.mlp.up_proj', layer.mlp.up_proj.weight.data.shape)
            # for i in range(layer.mlp.up_proj.weight.data.shape[0]):
            #     if i < 20:
            #         print('after pruning layer.mlp.up_proj', layer.mlp.up_proj.weight.data[i, :100])
            #     else:
            #         break
            # print('layer.mlp.up_proj.weight.data.shape', layer.mlp.gate_proj.weight.data.shape, torch.where(mlp_mask)[0].shape)
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
            
            # Update output dimensions of up and gate projections based on the mlp mask
            layer.mlp.up_proj.out_features = mlp_mask.sum().item()
            layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
            
            output_weight = layer.mlp.down_proj.weight.data
            layer.mlp.intermediate_size = mlp_mask.sum().item()
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
              
            # Prune the down projection weight
            output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]  
            
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                layer.mlp.down_proj.in_features = mlp_mask.sum().item()
                # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                layer.mlp.down_proj.bias = nn.Parameter(torch.zeros(output_bias.shape, device=device))
                layer.mlp.down_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.mlp.down_proj.weight.data = output_weight

            layer.mlp.up_proj.is_pruned = True
            layer.mlp.gate_proj.is_pruned = True
            layer.mlp.down_proj.is_pruned = True

            layer.mlp.up_proj.key = f'{idx}.mlp.up_proj'
            layer.mlp.gate_proj.key = f'{idx}.mlp.gate_proj'
            layer.mlp.down_proj.key = f'{idx}.mlp.down_proj'
    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()
    
def update_pruning_info(logger_into, info):
    logger_into.update(info)
    return

def cal_remove_neuron(args, model):
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    if hardcode_struct == "UL-MM":
        remove_params = cfg['prune_hyper'] * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * (args.remove_heads // num_layers) * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))
    else:
        remove_params = num_layers * cfg['prune_hyper'] * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * args.remove_heads * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))

def prune_flap_llama(model, tokenizer, dataloader, logger_info, device=torch.device("cuda:0")):
    """
    Our FLAP Pruning.
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    mysetting = MySettings()
    hardcode_struct = 'UL-UM'
    if 'global' in cfg['prune_name']:
        hardcode_struct = 'AL-AM'
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    # print("loading calibdation data")
    # dataloader, _ = get_loaders("wikitext2", nsamples=cfg["nsamples"],seed=args.seed,seqlen=cfg[cfg['model_name']]['max_length'],tokenizer=tokenizer)
    # print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    layers = model.model.layers

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []

    target_modules = _get_target_modules(cfg)
    layers = model.model.layers

    dev = device
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        key_list = [key for key, _ in layer.named_modules()]
        # print('key_list', key_list)
        for key in key_list:
            if not _check_target_module_exists(target_modules, key):
                continue

            # print('found_pruning_layers', key, flush=True)
            _, target, _ = _get_submodules(layer, key)
            subset[key] = target

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = subset[name]     

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].get_pre_hook()(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(cfg["nsamples"]):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if name == 'self_attn.o_proj':
                W_metric = metrics[cfg['prune_metric']](wrapped_layers, subset, name)
                # flap's manual trick, attention square the metric
                if 'square' in cfg['prune_name']:
                    W_metric = W_metric ** 2
                W_metric = metric_process(mysetting, W_metric)
                if hardcode_struct == "UL-UM":
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][int(cfg['prune_hyper']*layer.self_attn.num_heads)].cpu()
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                    compress(i,layer, W_mask, None, None, None, dev, bias=mysetting.add_bias, unstr=False)
                elif hardcode_struct == "UL-MM":
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][args.remove_heads // len(layers)].cpu()
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                else:
                    attn_metric_list.append(W_metric.cpu())
                attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            else:
                W_metric = metrics[cfg['prune_metric']](wrapped_layers, subset, name)
                # flap's trick, MLP use the formula in paper
                W_metric = metric_process(mysetting, W_metric)
                if hardcode_struct == "UL-UM":
                    thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*cfg['prune_hyper'])].cpu()
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                    compress(i, layer, None, W_mask, None, None, dev, bias=mysetting.add_bias, unstr=False)
                elif hardcode_struct == "UL-MM":
                    thresh = torch.sort(W_metric.cuda())[0][cal_remove_neuron(args, model)].cpu()
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                else:
                    mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            wrapped_layers[name].free()


        for j in range(cfg["nsamples"]):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]        
        inps, outs = outs, inps # Use the original output as input to the next layer
        torch.cuda.empty_cache()

    if hardcode_struct in ["AL-MM", "AL-AM"]:
        if len(attn_metric_list) > 0:
            attn_metric = torch.stack(attn_metric_list)
            attn_metric = metric_process(mysetting, attn_metric)
            attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)
            print('attn_metric', attn_metric.shape, attn_metric)
        else:
            attn_metric = None

        # Check if len(mlp_metric_list) > 0 is not empty and process
        if len(mlp_metric_list) > 0:
            mlp_metric = torch.stack(mlp_metric_list)
            mlp_metric = metric_process(mysetting, mlp_metric)
            print('mlp_metric', mlp_metric.shape, mlp_metric)
        else:
            mlp_metric = None

        # Concatenate the metrics, handling cases where one or both may be None
        if attn_metric is not None and mlp_metric is not None:
            prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
        elif attn_metric is not None:
            prune_metric = attn_metric.view(-1)
        elif mlp_metric is not None:
            prune_metric = mlp_metric.view(-1)

        if hardcode_struct == "AL-MM":
            sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
            attn_thres = sorted_attn[-int(args.remove_heads)]
            attn_mask = (attn_metric > attn_thres)  # 1 means retain
            
            sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
            mlp_thres = sorted_mlp[-cal_remove_neuron(args, model)]
            mlp_mask = (mlp_metric > mlp_thres)
        else:
            # prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
            print('prune_metric', prune_metric.shape, prune_metric)
            sorted_prune, indices = torch.sort(prune_metric)
            print(', sorted_prune.shape[0]', sorted_prune.numel())
            # compression_weight = torch.ones_like(indices)
            # compression_weight[indices < attn_metric.numel()] = 512.0 / 3

            # print('attn_metric.numel()', attn_metric.numel())
            # print('mlp_metric.numel()', mlp_metric.numel())
            # print('compression_weight', compression_weight.shape, compression_weight, torch.sum(compression_weight), torch.sum(compression_weight)*(1 - cfg['prune_hyper']) )
            # print('zzz', torch.abs( torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*(1 - cfg['prune_hyper']) ))
            # print('final index', torch.argmin(torch.abs( torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*(1 - cfg['prune_hyper']) )))
            # threshold = sorted_prune[torch.argmin(torch.abs( torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*(1 - cfg['prune_hyper']) ))]
            threshold = sorted_prune[int(sorted_prune.numel() * cfg['prune_hyper'])]
            if len(attn_metric_list) > 0:
                attn_mask = (attn_metric > threshold)
            if len(mlp_metric_list) > 0:
                mlp_mask = (mlp_metric > threshold)
    else:
        if len(attn_mask) > 0:
            attn_mask = torch.stack(attn_mask) 
        if len(mlp_mask) > 0:
            mlp_mask = torch.stack(mlp_mask)
    

    if mysetting.global_prune:
        for idx in range(len(layers)):
            if len(attn_metric_list) > 0:
                if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
                    compress(idx, model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, model.hf_device_map[f"model.layers.{idx}"], bias=mysetting.add_bias, unstr=False)
                else:
                    compress(idx, model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, device, bias=mysetting.add_bias, unstr=False)

            if len(mlp_metric_list) > 0:
                if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
                    compress(idx,model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], model.hf_device_map[f"model.layers.{idx}"], bias=mysetting.add_bias, unstr=False)
                else:
                    compress(idx,model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], device, bias=mysetting.add_bias, unstr=False)
            # compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def parallel_cal_varying_length_norm(sorted_norm, norm):
    # sorted_norm is non-negative
    processed_channels = sorted_norm.pow(norm)
    # print('processed_channels', processed_channels.shape, processed_channels[0])
    varying_vector_norm = torch.pow(processed_channels.cumsum(dim=0), 1/norm)
    return varying_vector_norm
            
def parallel_cal_varying_length_info(sorted_norm, pq_p, pq_q, reversed=False):
    if reversed:
        sorted_norm = torch.flip(sorted_norm, [1])
    nominator_varying_vector_norm = parallel_cal_varying_length_norm(sorted_norm, pq_p)
    denominator_varying_vector_norm = parallel_cal_varying_length_norm(sorted_norm, pq_q)

    nominator_varying_vector_norm = nominator_varying_vector_norm.to(cfg['device'])
    denominator_varying_vector_norm = denominator_varying_vector_norm.to(cfg['device'])
    # print('nominator_varying_vector_norm', nominator_varying_vector_norm.shape, nominator_varying_vector_norm[0])
    # print('denominator_varying_vector_norm', denominator_varying_vector_norm.shape, denominator_varying_vector_norm[0])

    # num_rows, num_cols = nominator_varying_vector_norm.shape

    # if reversed:
    #     # Create a tensor where each row starts from 1 and decreases to the length of the row
    #     dimension = torch.arange(num_cols, 0, -1).unsqueeze(0)
    # else:
        # Create a tensor where each row starts from 1 and increases to the length of the row
    dimension = torch.arange(1, nominator_varying_vector_norm.shape[0] + 1).to(cfg['device'])
    # dimension = dimension.expand(nominator_varying_vector_norm.shape[0], -1).to(cfg['device'])
    return nominator_varying_vector_norm, denominator_varying_vector_norm, dimension

def cal_prune_count_base_on_pq(sorted_tensor, pq_p, pq_q, eta, pq_beta, pq_gamma):

    # norm_across_other_dims = norm_across_other_dims + (norm_across_other_dims == 0) * 1e-9
    # Calculate norms only for non-zero channels
    # non_zero_norms = norm_across_other_dims[non_zero_mask]

    # norm_p = torch.linalg.vector_norm(sorted_tensor, ord=pq_p, dim=0)
    # norm_q = torch.linalg.vector_norm(sorted_tensor, ord=pq_q, dim=0) + 1e-10
    
    # dimension = sorted_tensor.shape[0]
    # pq_indices = (1 - dimension ** (1/pq_q - 1/pq_p) * (norm_p / norm_q))
    
    # # add additional dimension if dimension is 0
    # # if pq_indices.dim() == 0 or pq_indices.dim() == 1:
    # #     pq_indices.unsqueeze_(0)
    # print('pq_indices', pq_indices, dimension)
    # if torch.isnan(pq_indices).any():
    #     pq_indices = torch.min(pq_indices, torch.ones_like(pq_indices))
    #     raise ValueError('pq_indices contains nan values')

    # lower_bound = dimension * (1 + eta) ** (-pq_q / (pq_q - pq_p)) * ((1 - pq_indices) ** (pq_q * pq_p / (pq_q - pq_p)))
    # print('lower_bound', lower_bound, dimension)
    # beta_tensor = torch.full_like(lower_bound, pq_beta)
    # prune_channels_count = torch.floor(dimension * torch.min(pq_gamma * (1 - lower_bound / dimension), beta_tensor))
    # print('prune_channels_count', prune_channels_count)

    # return int(prune_channels_count), pq_indices

    nominator_varying_vector_norm, denominator_varying_vector_norm, dimension = parallel_cal_varying_length_info(sorted_tensor, pq_p, pq_q)
    pq_indices_varying_length = (1 - dimension ** (1/pq_q - 1/pq_p) * (nominator_varying_vector_norm / denominator_varying_vector_norm))
    lower_bound = dimension * (1 + eta) ** (-pq_q / (pq_q - pq_p)) * ((1 - pq_indices_varying_length) ** (pq_q * pq_p / (pq_q - pq_p)))
    
    # lower_bound = lower_bound.cpu().numpy()
    # x = list(range(len(lower_bound.tolist())))
    # dx = np.diff(x)
    # dy = np.diff(lower_bound)

    # # Compute slope
    # slopes = dy / dx
    
    # if 'low' in cfg['prune_name']:
    #     # avoid edge case of slope
    #     window_size = 21  # 10 neighbors on each side + the element itself

    #     # Create a window with equal weights
    #     window = np.ones(window_size) / window_size
    #     # Calculate the moving average using convolution
    #     averages = np.convolve(slopes, window, 'same')
    #     abs_averages_slopes = np.abs(averages)
    #     # Find the index of the minimum value in abs_slopes
    #     first_phase_transition = np.argmin(abs_averages_slopes)
    #     pq_indices = pq_indices_varying_length[first_phase_transition]
    #     lower_bound = lower_bound[first_phase_transition]
    x = torch.arange(len(lower_bound), dtype=torch.float32, device=lower_bound.device)

    # Calculate differences (equivalent to np.diff)
    dx = x[1:] - x[:-1]
    dy = lower_bound[1:] - lower_bound[:-1]

    # Compute slope
    slopes = dy / dx

    if 'low' in cfg['prune_name']:
        # Avoid edge case of slope, just randomly pick this number
        window_size = 20  # 10 neighbors on each side + the element itself

        # Create a window with equal weights
        window = torch.ones(window_size, dtype=torch.float32,  device=lower_bound.device) / window_size
        window = window.to(lower_bound.device)  # Ensure window is on the same device as lower_bound

        # Calculate the moving average using convolution
        # PyTorch's conv1d expects a 3D tensor (batch, channel, length), so we need to add extra dimensions
        slopes = slopes.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        window = window.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Use conv1d for moving average
        averages = torch.nn.functional.conv1d(slopes, window, padding=window_size//2)
        averages = averages.squeeze()  # Remove extra dimensions

        negative_values = averages[averages <= 0]

        # Check if there are any negative values
        if len(negative_values) > 0:
            # Find the maximum among the negative values (closest to zero)
            closest_negative = torch.max(negative_values)

            # Get the index of this value in the original 'averages' tensor
            first_phase_transition = torch.where(averages == closest_negative)[0][0]
        else:
            first_phase_transition = None  # or handle the case where there are no negative values
            raise ValueError('No negative values found in averages')

        print("Index of negative value closest to zero:", first_phase_transition, sorted_tensor.shape[0])
        pq_indices = pq_indices_varying_length[first_phase_transition]
        lower_bound = lower_bound[first_phase_transition]
        dimension = dimension[first_phase_transition]
    elif 'high' in cfg['prune_name']:
        slopes = torch.abs(dy / dx)
        threshold = 0.05 * slopes.shape[0]
        indices = torch.where(slopes > threshold)[0]
        if len(indices) > 0:
            second_phase_transition = indices[0].item()  # Get the first index as a Python scalar
        else:
            print('dont find second phase transition')
            second_phase_transition = lower_bound.shape[0] - 1  # or handle the case where there are no negative values

        pq_indices = pq_indices_varying_length[second_phase_transition]
        lower_bound = lower_bound[second_phase_transition]
        dimension = dimension[second_phase_transition]
        print("Index of negative value closest to zero:", second_phase_transition, sorted_tensor.shape[0])
    else:
        pq_indices = pq_indices_varying_length[-1]
        lower_bound = lower_bound[-1]
        dimension = dimension[-1]
        print("Index of negative value closest to zero:", pq_indices, dimension, sorted_tensor.shape[0])

    beta_tensor = torch.full_like(lower_bound, pq_beta)
    prune_channels_count = torch.floor(dimension * torch.min(pq_gamma * (1 - lower_bound / dimension), beta_tensor))

    prune_channels_count = prune_channels_count.to(cfg['device'])
    pq_indices = pq_indices_varying_length.to(cfg['device'])
    return int(prune_channels_count), pq_indices


def prune_pq_llama(model, tokenizer, dataloader, logger_info, device=torch.device("cuda:0")):
    """
    pq on structured pruning, no bias recover.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    mysetting = MySettings()
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    pq_p = cfg['pq_p']
    pq_q = cfg['pq_q']
    eta = cfg['prune_hyper']
    pq_beta = cfg['pq_beta']
    pq_gamma = cfg['pq_gamma']
    # print("loading calibdation data")
    # dataloader, _ = get_loaders("c4",nsamples=128,seed=args.seed,seqlen=cfg[cfg['model_name']]['max_length'],tokenizer=tokenizer)
    # print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    attn_metric_list, mlp_metric_list = [], []

    target_modules = _get_target_modules(cfg)
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        key_list = [key for key, _ in layer.named_modules()]
        # print('key_list', key_list)
        for key in key_list:
            if not _check_target_module_exists(target_modules, key):
                continue

            # print('found_pruning_layers', key, flush=True)
            _, target, _ = _get_submodules(layer, key)
            subset[key] = target
        
        dev = device
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = subset[name]     

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].get_pre_hook()(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(cfg["nsamples"]):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = metrics[cfg['prune_metric']](wrapped_layers, subset, name)
            W_metric = metric_process(mysetting, W_metric)

            # print('W_metric',  W_metric, type(W_metric))
            # print(W_metric.shape)
            logger_key = f"layer{i}_{name}"
            if name == 'self_attn.o_proj':
                if mysetting.global_prune:
                    attn_metric_list.append(W_metric.cpu())
                else:
                    if 'normhead' in cfg['prune_name']:
                        reshaped_W_metric = W_metric.reshape(-1, 128)
                        # Calculate the L2 norm for each 128-element vector
                        W_metric = torch.norm(reshaped_W_metric, p=2, dim=1)
                    else:
                        W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    # print('W_metric2', W_metric.shape, W_metric, type(W_metric))
                    prune_count, pq_indices = cal_prune_count_base_on_pq(torch.sort(W_metric.cuda())[0], pq_p, pq_q, eta, pq_beta, pq_gamma)
                    print('atten', torch.sort(W_metric.cuda())[0], W_metric.tolist(), prune_count)
                    thresh = torch.sort(W_metric.cuda())[0][prune_count].cpu()
                    W_mask = (W_metric>=thresh)
                    compress(i,layer, W_mask, None, None, None, dev, bias=mysetting.add_bias, unstr=False)

                    nominator_varying_vector_norm, denominator_varying_vector_norm, dimension = parallel_cal_varying_length_info(torch.sort(W_metric.cuda())[0], pq_p, pq_q)
                    # print('dimension', dimension.shape, dimension)
                    pq_indices_varying_length = (1 - dimension ** (1/pq_q - 1/pq_p) * (nominator_varying_vector_norm / denominator_varying_vector_norm))

                    info = {
                        f'{logger_key}_norm_across_other_dims': W_metric.tolist(),
                        f'{logger_key}_pq_indices_varying_lengths': pq_indices_varying_length.tolist(),
                        f'{logger_key}_pq_lower_bound': prune_count,
                        f'{logger_key}_pq_indices': pq_indices.tolist()
                    }
                    update_pruning_info(logger_info, info)
                    # print('attnW_metric.tolist()', W_metric.tolist())
                    # print('attnpq_indices_varying_length.tolist()',  pq_indices_varying_length.tolist())
            else:
                if mysetting.global_prune:
                    mlp_metric_list.append(W_metric.cpu())
                else:
                    print('mlpW_metric', W_metric.shape, W_metric)
                    # a = (torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1))).reshape(-1, 1) * torch.linalg.vector_norm(subset[name].weight.data, ord=1, dim=1).reshape(1, -1))
                    # print('mlpW_metric2', a.shape, a)
                    # b = torch.sum(a, dim=1)
                    # print('mlpW_metric3', b.shape, b)
                    sorted_prune = torch.sort(W_metric.cuda())[0]
                    prune_count, pq_indices = cal_prune_count_base_on_pq(sorted_prune, pq_p, pq_q, eta, pq_beta, pq_gamma)
                    # print('mlp', sorted_prune, sorted_prune.shape, prune_count)
                    thresh = sorted_prune[prune_count].cpu()
                    W_mask = (W_metric>=thresh)
                    # print('W_mask', W_mask.shape, )
                    compress(i,layer, None, W_mask, None, None, dev, bias=mysetting.add_bias, unstr=False)

                    nominator_varying_vector_norm, denominator_varying_vector_norm, dimension = parallel_cal_varying_length_info(torch.sort(W_metric.cuda())[0], pq_p, pq_q)
                    # print('dimension', dimension.shape, dimension)
                    pq_indices_varying_length = (1 - dimension ** (1/pq_q - 1/pq_p) * (nominator_varying_vector_norm / denominator_varying_vector_norm))
                    info = {
                        f'{logger_key}_norm_across_other_dims': W_metric.tolist(),
                        f'{logger_key}_pq_indices_varying_lengths': pq_indices_varying_length.tolist(),
                        f'{logger_key}_pq_lower_bound': prune_count,
                        f'{logger_key}_pq_indices': pq_indices.tolist()
                    }
                    update_pruning_info(logger_info, info)
                    # print('mlpW_metric.tolist()', W_metric.tolist())
                    # print(' mlppq_indices_varying_length.tolist()',  pq_indices_varying_length.tolist())
                    
            wrapped_layers[name].free()


        for j in range(cfg["nsamples"]):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # the pruned output as input to the next layer
        
        torch.cuda.empty_cache()

    if mysetting.global_prune:
        if len(attn_metric_list) > 0:
            attn_metric = torch.stack(attn_metric_list)
            attn_metric = metric_process(mysetting, attn_metric)
            if 'normhead' in cfg['prune_name']:
                attn_metric = attn_metric.reshape(len(layers), -1, 128)
                attn_metric = torch.norm(attn_metric, p=2, dim=2)
            else:
                attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)
        else:
            attn_metric = None

        # Check if len(mlp_metric_list) > 0 is not empty and process
        if len(mlp_metric_list) > 0:
            mlp_metric = torch.stack(mlp_metric_list)
            mlp_metric = metric_process(mysetting, mlp_metric)
            for i in range(mlp_metric.shape[0]):
                print('mlp_metric for each layer', i, mlp_metric[i].shape, mlp_metric[i], torch.sort(mlp_metric[i])[0])
        else:
            mlp_metric = None

        # Concatenate the metrics, handling cases where one or both may be None
        if attn_metric is not None and mlp_metric is not None:
            prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
        elif attn_metric is not None:
            prune_metric = attn_metric.view(-1)
        elif mlp_metric is not None:
            prune_metric = mlp_metric.view(-1)

        sorted_prune, indices = torch.sort(prune_metric)
        print('sorted_prune', sorted_prune.shape, sorted_prune)
        prune_count, pq_indices = cal_prune_count_base_on_pq(sorted_prune, pq_p, pq_q, eta, pq_beta, pq_gamma)
        
        threshold = sorted_prune[prune_count]
        if len(attn_metric_list) > 0:
            attn_mask = (attn_metric > threshold)
        if len(mlp_metric_list) > 0:
            mlp_mask = (mlp_metric > threshold)

        nominator_varying_vector_norm, denominator_varying_vector_norm, dimension = parallel_cal_varying_length_info(torch.sort(sorted_prune.cuda())[0], pq_p, pq_q)
        # print('dimension', dimension.shape, dimension)
        pq_indices_varying_length = (1 - dimension ** (1/pq_q - 1/pq_p) * (nominator_varying_vector_norm / denominator_varying_vector_norm))
        print('pq_indices_varying_length', pq_indices_varying_length)
        info = {
            f'global_norm_across_other_dims': sorted_prune.tolist(),
            f'global_pq_indices_varying_lengths': pq_indices_varying_length.tolist(),
            f'global_pq_lower_bound': prune_count,
            f'global_pq_indices': pq_indices.tolist()
        }
        update_pruning_info(logger_info, info)
        for idx in range(len(layers)):
            if len(attn_metric_list) > 0:
                if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
                    compress(idx,model.model.layers[idx], attn_mask[idx], None, None, None, model.hf_device_map[f"model.layers.{idx}"], bias=mysetting.add_bias, unstr=False)
                else:
                    compress(idx,model.model.layers[idx], attn_mask[idx], None, None, None, device, bias=mysetting.add_bias, unstr=False)
            
            if len(mlp_metric_list) > 0:
                if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
                    compress(idx,model.model.layers[idx], None, mlp_mask[idx], None, None, model.hf_device_map[f"model.layers.{idx}"], bias=mysetting.add_bias, unstr=False)
                else:
                    compress(idx,model.model.layers[idx], None, mlp_mask[idx], None, None, device, bias=mysetting.add_bias, unstr=False)
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_wanda_sp_llama(model, tokenizer, dataloader, logger_info, device=torch.device("cuda:0")):
    """
    Wanda on structured pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    mysetting = MySettings()
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    target_modules = _get_target_modules(cfg)
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        key_list = [key for key, _ in layer.named_modules()]
        for key in key_list:
            if not _check_target_module_exists(target_modules, key):
                continue

            print('found_pruning_layers', key, flush=True)
            _, target, _ = _get_submodules(layer, key)
            subset[key] = target

        dev = device
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = subset[name]     

        def add_batch(name):
            def tmp(_, inp, out):
                # print(f"{name} tmp inp", inp)
                wrapped_layers[name].get_pre_hook()(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(cfg["nsamples"]):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = metrics[cfg['prune_metric']](wrapped_layers, subset, name)
            W_metric = metric_process(mysetting, W_metric)         

            # test ratio
            # if 'gate_proj' in name:
            #     temp = subset[name].baseline_inp
            #     print('self.baseline_inp', subset[name].baseline_inp.shape, outs.shape, outs.mean(dim=(0,1)).shape, subset[name].baseline_inp)
            #     temp_out = outs.mean(dim=(0,1)) - temp
            #     print('temp_out', temp_out.shape, temp_out, temp_out.mean())
            #     ratio = temp / temp_out
            #     print('ratio', outs.mean(dim=0).shape, temp_out.shape, ratio.shape, ratio, ratio.mean())
            #     continue

            if name == 'self_attn.o_proj':
                if mysetting.global_prune:
                    attn_metric_list.append(W_metric.cpu())
                else:
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)    # importance score of each head
                    print('attnW_metric', W_metric.shape, W_metric)
                    thresh = torch.sort(W_metric.cuda())[0][int(cfg['prune_hyper']*layer.self_attn.num_heads)].cpu()
                    W_mask = (W_metric>=thresh)
                    compress(i,layer, W_mask, None, None, None, dev, bias=False, unstr=False)
                attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            else:
                if mysetting.global_prune:
                    mlp_metric_list.append(W_metric.cpu())
                else:
                    print('mlpW_metric', W_metric.shape, W_metric, int(W_metric.numel()))
                    thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*cfg['prune_hyper'])].cpu()
                    W_mask = (W_metric>=thresh)
                    compress(i, layer, None, W_mask, None, None, dev, bias=False, unstr=False)
                mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
                # compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
            wrapped_layers[name].free()

        for j in range(cfg["nsamples"]):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # the pruned output as input to the next layer
        
        torch.cuda.empty_cache()

    if mysetting.global_prune:
        if len(attn_metric_list) > 0:
            attn_metric = torch.stack(attn_metric_list)
            attn_metric = metric_process(mysetting, attn_metric)
            attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)
        else:
            attn_metric = None

        # Check if len(mlp_metric_list) > 0 is not empty and process
        if len(mlp_metric_list) > 0:
            mlp_metric = torch.stack(mlp_metric_list)
            mlp_metric = metric_process(mysetting, mlp_metric)
        else:
            mlp_metric = None

        # Concatenate the metrics, handling cases where one or both may be None
        if attn_metric is not None and mlp_metric is not None:
            print('both not none')
            prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
        elif attn_metric is not None:
            prune_metric = attn_metric.view(-1)
        elif mlp_metric is not None:
            prune_metric = mlp_metric.view(-1)
        sorted_prune, indices = torch.sort(prune_metric)
        threshold = sorted_prune[int(sorted_prune.numel() * cfg['prune_hyper'])]
        print('threshold', threshold)
        if len(attn_metric_list) > 0:
            attn_mask = (attn_metric > threshold)
            print('attn_mask', attn_mask.shape, attn_mask)
        if len(mlp_metric_list) > 0:
            mlp_mask = (mlp_metric > threshold)
            print('mlp_mask', mlp_mask.shape, mlp_mask)
        for idx in range(len(layers)):
            if len(attn_metric_list) > 0:
                if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
                    compress(idx,model.model.layers[idx], attn_mask[idx], None,  attn_baseline_inp_list[idx], None, model.hf_device_map[f"model.layers.{idx}"], bias=mysetting.add_bias, unstr=False)
                else:
                    compress(idx,model.model.layers[idx], attn_mask[idx], None,  attn_baseline_inp_list[idx], None, device, bias=mysetting.add_bias, unstr=False)
            
            if len(mlp_metric_list) > 0:
                if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
                    compress(idx,model.model.layers[idx], None, mlp_mask[idx], None,  mlp_baseline_inp_list[idx], model.hf_device_map[f"model.layers.{idx}"], bias=mysetting.add_bias, unstr=False)
                else:
                    compress(idx,model.model.layers[idx], None, mlp_mask[idx], None,  mlp_baseline_inp_list[idx], device, bias=mysetting.add_bias, unstr=False)
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    
def prune_magnitude_sp_llama(model, tokenizer, dataloader, device=torch.device("cuda:0")):
    """
    Magnitude Pruning on structured pruning.
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    add_bias = if_add_bias()
    standardize = if_standardize()
    global_prune = if_global_prune()
    layers = model.model.layers 
    target_modules = _get_target_modules(cfg)
    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        key_list = [key for key, _ in layer.named_modules()]
        # print('key_list', key_list)
        for key in key_list:
            if not _check_target_module_exists(target_modules, key):
                continue

            print('found_pruning_layers', key, flush=True)
            _, target, _ = _get_submodules(layer, key)
            subset[key] = target

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
            device = model.hf_device_map[f"model.layers.{i}"]
        
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.norm(subset[name].weight.data, dim=0)
            if 'std' in cfg['prune_name']:
                W_metric = standarlization(W_metric)
            if name == 'self_attn.o_proj':
                W_metric = W_metric.reshape(-1, 128).sum(dim=1) # importance score of each head
                thresh = torch.sort(W_metric.cuda())[0][int(cfg['prune_hyper']*layer.self_attn.num_heads)].cpu()
                W_mask = (W_metric>=thresh)
                compress(i,layer, W_mask, None, None, None, device, bias=False, unstr=False)
            else:
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*cfg['prune_hyper'])].cpu()
                W_mask = (W_metric>=thresh)
                compress(i,layer, None, W_mask, None, None, device, bias=False, unstr=False)
            