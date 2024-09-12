import torch
from config import cfg
from module import nearest_multiple, check_skip_layers


class HiddenRepresentationPruning():

    def __init__(self, cfg, key, model_config=None):
        self.key = key
        self.prune_metric = cfg['prune_metric']
        # if flap ratio in prune method, will update later
        if not isinstance(cfg['prune_ratio'], float) and len(cfg['prune_ratio']) == 2:
            if 'attention' in self.key:
                self.prune_ratio = self.adjust_prune_ratio(cfg['prune_ratio'][0], model_config)
            else:
                self.prune_ratio = self.adjust_prune_ratio(cfg['prune_ratio'][1], model_config)
        else:
            self.prune_ratio = self.adjust_prune_ratio(cfg['prune_ratio'], model_config)
        self.model_config = model_config

    def adjust_prune_ratio(self, prune_ratio, model_config):
        if model_config is not None:
            if prune_ratio == 0:
                return 0
            
            num_hidden_layers = model_config.num_hidden_layers
            # TODO: command back
            prune_ratio = round(num_hidden_layers / (num_hidden_layers - (len(cfg['skip_layers']))) * prune_ratio, 2) 
            # prune_ratio = cfg['prune_ratio']
            # if ('llama-3' in cfg['model_name'] or 'llama-2-70b' in cfg['model_name']) and ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']) :
            #     if 'csr' in cfg['task_name']:
            #         approx_seq_len = 100
            #     elif 'clm' in cfg['task_name']:
            #         approx_seq_len = cfg['max_seq_len']
            #     # since we dont prune kv, increase the pruning ratio a little bit
            #     head_dim = model_config.hidden_size // model_config.num_attention_heads
            #     num_key_value_groups = model_config.num_attention_heads // model_config.num_key_value_heads
                
            #     attn_flops = approx_seq_len * model_config.hidden_size * model_config.hidden_size * 2 + approx_seq_len * model_config.hidden_size * model_config.hidden_size//num_key_value_groups * 2 + model_config.num_attention_heads * approx_seq_len ** 2 * head_dim + model_config.num_attention_heads * approx_seq_len ** 2 * head_dim
            #     mlp_flops = approx_seq_len * model_config.hidden_size * model_config.intermediate_size * 3
            #     cur_layer_total_flops = attn_flops + mlp_flops

            #     prune_heads = int(prune_ratio * model_config.num_attention_heads)
            #     prune_attn_dimension = prune_heads * head_dim
            #     prune_mlp_dimension = int(prune_ratio * model_config.intermediate_size)

            #     prune_attn_flops = approx_seq_len * model_config.hidden_size * prune_attn_dimension * 2 + prune_heads * approx_seq_len ** 2 * head_dim + prune_heads * approx_seq_len ** 2 * head_dim
            #     prune_mlp_flops = approx_seq_len * model_config.hidden_size * prune_mlp_dimension * 3
            #     prune_layer_total_flops = prune_attn_flops + prune_mlp_flops

            #     multiple_ratio = (cur_layer_total_flops * prune_ratio / prune_layer_total_flops)
            #     prune_ratio = prune_ratio * (cur_layer_total_flops * prune_ratio / prune_layer_total_flops)
            return prune_ratio
        else:
            return cfg['prune_ratio']
    
    def handle_pad_tokens(self, probe_out, bsz_selected_indices, seq_selected_indices):
        # handling padding tokens
        # tbh, it is a little annoying for padding tokens
        if cfg['pad_tokens'] is not None:
            cfg['pad_tokens'] = cfg['pad_tokens'].to(probe_out.device)
            if bsz_selected_indices is not None:
                probe_pad_tokens = cfg['pad_tokens'][bsz_selected_indices]
            else:
                probe_pad_tokens = cfg['pad_tokens']

            if seq_selected_indices is not None:
                probe_pad_tokens = probe_pad_tokens[:, seq_selected_indices]

            probe_out[probe_pad_tokens] = 0
            probe_num = (probe_out.size(0) - torch.sum(probe_pad_tokens, dim=0) + 1e-3).to(probe_out.dtype).unsqueeze(1)
            return probe_out, probe_num
        else:
            return probe_out, probe_out.size(0)
        

    def cal_attn_prune_metric(self, probe_out, weight, metric_type, bsz_selected_indices, seq_selected_indices, global_metric_score_distribution=None):
        probe_out, probe_num = self.handle_pad_tokens(probe_out, bsz_selected_indices, seq_selected_indices)
        # if 'ppwandaspsum' in metric_type:
        #     combined_probe_out = None
        #     if global_metric_score_distribution is not None:
        #         cur_global_metric_score_distribution = global_metric_score_distribution[seq_selected_indices, :] if seq_selected_indices is not None else global_metric_score_distribution
        #         cur_global_metric_score_distribution = cur_global_metric_score_distribution.to(probe_out.device)
        #         norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])

        #         if 'probefixratio' in cfg['prune_method']:
        #             combined_probe_out = cfg['probefixratio'] * cur_global_metric_score_distribution + (1-cfg['probefixratio']) * norm_probe_out_square
        #         # dynaratio, since all nonnegative, no need to abs
        #         else:  
        #             denominator = norm_probe_out_square + cur_global_metric_score_distribution
        #             # avoid nan, nan is always a problem in float16
        #             # tend to give the global metric more weight
        #             probe_ratio = norm_probe_out_square / (denominator + 1e-6)
        #             global_ratio = 1 - probe_ratio
        #             combined_probe_out = global_ratio * cur_global_metric_score_distribution + probe_ratio * norm_probe_out_square

        #         combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
        #         probe_out_dim_metric = torch.linalg.vector_norm((combined_probe_out.reshape((1,-1)) * torch.pow(weight, 2)).reshape(self.model_config.hidden_size, self.model_config.num_attention_heads, -1), ord=2, dim=(0, 2)).clamp(max=cfg['data_type_max'])
        #         return probe_out_dim_metric
        #     else:
        #         norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])
        #         norm_probe_out_square = torch.sum(norm_probe_out_square, dim=0).clamp(max=cfg['data_type_max'])
        #         probe_out_dim_metric = torch.linalg.vector_norm((norm_probe_out_square.reshape((1,-1)) * torch.pow(weight, 2)).reshape(self.model_config.hidden_size, self.model_config.num_attention_heads, -1), ord=2, dim=(0, 2)).clamp(max=cfg['data_type_max'])
        #         return probe_out_dim_metric
        if 'ppwandasp' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                cur_global_metric_score_distribution = global_metric_score_distribution[seq_selected_indices, :] if seq_selected_indices is not None else global_metric_score_distribution
                cur_global_metric_score_distribution = cur_global_metric_score_distribution.to(probe_out.device)
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])

                if 'probefixratio' in cfg['prune_method']:
                    combined_probe_out = cfg['probefixratio'] * cur_global_metric_score_distribution + (1-cfg['probefixratio']) * norm_probe_out_square
                # dynaratio, since all nonnegative, no need to abs
                else:  
                    denominator = norm_probe_out_square + cur_global_metric_score_distribution
                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight
                    probe_ratio = norm_probe_out_square / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * cur_global_metric_score_distribution + probe_ratio * norm_probe_out_square

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = torch.linalg.vector_norm((combined_probe_out.reshape((1,-1)) * torch.pow(weight, 2)).reshape(self.model_config.hidden_size, self.model_config.num_attention_heads, -1), ord=2, dim=(0, 2)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            else:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])
                norm_probe_out_square = torch.sum(norm_probe_out_square, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = torch.linalg.vector_norm((norm_probe_out_square.reshape((1,-1)) * torch.pow(weight, 2)).reshape(self.model_config.hidden_size, self.model_config.num_attention_heads, -1), ord=2, dim=(0, 2)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
        elif 'wandasp' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                cur_global_metric_score_distribution = global_metric_score_distribution[seq_selected_indices, :] if seq_selected_indices is not None else global_metric_score_distribution
                cur_global_metric_score_distribution = cur_global_metric_score_distribution.to(probe_out.device)
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])

                if 'probefixratio' in cfg['prune_method']:
                    combined_probe_out = cfg['probefixratio'] * cur_global_metric_score_distribution + (1-cfg['probefixratio']) * norm_probe_out_square
                # dynaratio, since all nonnegative, no need to abs
                else:  
                    denominator = norm_probe_out_square + cur_global_metric_score_distribution
                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight
                    probe_ratio = norm_probe_out_square / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * cur_global_metric_score_distribution + probe_ratio * norm_probe_out_square

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = (torch.sqrt(combined_probe_out.reshape((1,-1))) * torch.abs(weight)).sum(dim=0).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            else:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])
                norm_probe_out_square = torch.sum(norm_probe_out_square, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = (torch.sqrt(norm_probe_out_square.reshape((1,-1))) * torch.abs(weight)).sum(dim=0).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
        elif 'flap' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                cur_global_metric_score_distribution = global_metric_score_distribution[seq_selected_indices, :] if seq_selected_indices is not None else global_metric_score_distribution
                cur_global_metric_score_distribution = cur_global_metric_score_distribution.to(probe_out.device)
                mean_probe_out = torch.mean(probe_out, dim=(0, 1))
                probe_variance = torch.sum(torch.pow(probe_out - mean_probe_out.reshape(1, 1, -1), 2), dim=0).clamp(max=cfg['data_type_max']) / probe_num

                if 'probefixratio' in cfg['prune_method']:
                    combined_probe_out = cfg['probefixratio'] * cur_global_metric_score_distribution + (1-cfg['probefixratio']) * probe_variance
                # dynaratio, since all nonnegative, no need to abs
                else:  
                    denominator = probe_variance + cur_global_metric_score_distribution
                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight
                    probe_ratio = probe_variance / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * cur_global_metric_score_distribution + probe_ratio * probe_variance

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = (combined_probe_out * torch.sum(torch.pow(weight, 2), dim=0)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            else:
                mean_probe_out = torch.mean(probe_out, dim=(0, 1), max=cfg['data_type_max'])
                probe_variance = torch.sum(torch.pow(probe_out - mean_probe_out.reshape(1, 1, -1), 2), dim=0).clamp(max=cfg['data_type_max']) / probe_num
                probe_out_dim_metric = (probe_variance * torch.sum(torch.pow(weight, 2), dim=0)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric

    
    def cal_mlp_prune_metric(self, probe_out, weight, metric_type, bsz_selected_indices, seq_selected_indices, global_metric_score_distribution=None):
        probe_out, probe_num = self.handle_pad_tokens(probe_out, bsz_selected_indices, seq_selected_indices)
        if 'ppwandasp' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                cur_global_metric_score_distribution = global_metric_score_distribution[seq_selected_indices, :] if seq_selected_indices is not None else global_metric_score_distribution
                cur_global_metric_score_distribution = cur_global_metric_score_distribution.to(probe_out.device)

                temp = torch.sum(cur_global_metric_score_distribution, dim=1)
                print('seq calibration', temp)
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])

                if 'probefixratio' in cfg['prune_method']:
                    combined_probe_out = cfg['probefixratio'] * cur_global_metric_score_distribution + (1-cfg['probefixratio']) * norm_probe_out_square
                # dynaratio, since all nonnegative, no need to abs
                else:  
                    denominator = norm_probe_out_square + cur_global_metric_score_distribution
                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight
                    probe_ratio = norm_probe_out_square / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * cur_global_metric_score_distribution + probe_ratio * norm_probe_out_square

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = torch.linalg.vector_norm((combined_probe_out.reshape((1,-1)) * torch.pow(weight, 2)), ord=2, dim=0).clamp(max=cfg['data_type_max'])
                # print('probe_out_dim_metric', probe_out_dim_metric.dtype)

                # if 'dimmetric' in cfg['prune_method']:
                #     cur_global_metric_score_distribution = global_metric_score_distribution[seq_selected_indices, :] if seq_selected_indices is not None else global_metric_score_distribution
                #     cur_global_metric_score_distribution = cur_global_metric_score_distribution.to(probe_out.device)
                #     cur_global_metric_score_distribution = torch.sum(cur_global_metric_score_distribution, dim=0)
                #     norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=(0, 1)) ** 2 / probe_num, max=cfg['data_type_max'])

                #     denominator = norm_probe_out_square + cur_global_metric_score_distribution
                #     # avoid nan, nan is always a problem in float16
                #     # tend to give the global metric more weight
                #     probe_ratio = norm_probe_out_square / (denominator + 1e-6)
                #     global_ratio = 1 - probe_ratio
                #     combined_probe_out = (global_ratio * cur_global_metric_score_distribution + probe_ratio * norm_probe_out_square).clamp(max=cfg['data_type_max'])
                #     probe_out_dim_metric = torch.linalg.vector_norm((combined_probe_out.reshape((1,-1)) * torch.pow(weight, 2)), ord=2, dim=0).clamp(max=cfg['data_type_max'])


                return probe_out_dim_metric
            else:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])
                norm_probe_out_square = torch.sum(norm_probe_out_square, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = torch.linalg.vector_norm((norm_probe_out_square.reshape((1,-1)) * torch.pow(weight, 2)), ord=2, dim=0).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
        elif 'wandasp' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                cur_global_metric_score_distribution = global_metric_score_distribution[seq_selected_indices, :] if seq_selected_indices is not None else global_metric_score_distribution
                cur_global_metric_score_distribution = cur_global_metric_score_distribution.to(probe_out.device)
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])

                if 'probefixratio' in cfg['prune_method']:
                    combined_probe_out = cfg['probefixratio'] * cur_global_metric_score_distribution + (1-cfg['probefixratio']) * norm_probe_out_square
                # dynaratio, since all nonnegative, no need to abs
                else:  
                    denominator = norm_probe_out_square + cur_global_metric_score_distribution
                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight
                    probe_ratio = norm_probe_out_square / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * cur_global_metric_score_distribution + probe_ratio * norm_probe_out_square

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = (torch.sqrt(combined_probe_out.reshape((1,-1))) * torch.abs(weight)).sum(dim=0).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            else:
                norm_probe_out_square = torch.clamp(torch.linalg.vector_norm(probe_out, ord=2, dim=0) ** 2 / probe_num, max=cfg['data_type_max'])
                norm_probe_out_square = torch.sum(norm_probe_out_square, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = (torch.sqrt(norm_probe_out_square.reshape((1,-1))) * torch.abs(weight)).sum(dim=0).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
        elif 'flap' in metric_type:
            combined_probe_out = None
            if global_metric_score_distribution is not None:
                cur_global_metric_score_distribution = global_metric_score_distribution[seq_selected_indices, :] if seq_selected_indices is not None else global_metric_score_distribution
                cur_global_metric_score_distribution = cur_global_metric_score_distribution.to(probe_out.device)
                mean_probe_out = torch.mean(probe_out, dim=(0, 1))
                probe_variance = torch.sum(torch.pow(probe_out - mean_probe_out.reshape(1, 1, -1), 2), dim=0).clamp(max=cfg['data_type_max']) / probe_num

                if 'probefixratio' in cfg['prune_method']:
                    combined_probe_out = cfg['probefixratio'] * cur_global_metric_score_distribution + (1-cfg['probefixratio']) * probe_variance
                # dynaratio, since all nonnegative, no need to abs
                else:  
                    denominator = probe_variance + cur_global_metric_score_distribution
                    # avoid nan, nan is always a problem in float16
                    # tend to give the global metric more weight
                    probe_ratio = probe_variance / (denominator + 1e-6)
                    global_ratio = 1 - probe_ratio
                    combined_probe_out = global_ratio * cur_global_metric_score_distribution + probe_ratio * probe_variance

                combined_probe_out = torch.sum(combined_probe_out, dim=0).clamp(max=cfg['data_type_max'])
                probe_out_dim_metric = (combined_probe_out * torch.sum(torch.pow(weight, 2), dim=0)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            else:
                mean_probe_out = torch.mean(probe_out, dim=(0, 1), max=cfg['data_type_max'])
                probe_variance = torch.sum(torch.pow(probe_out - mean_probe_out.reshape(1, 1, -1), 2), dim=0).clamp(max=cfg['data_type_max']) / probe_num
                probe_out_dim_metric = (probe_variance * torch.sum(torch.pow(weight, 2), dim=0)).clamp(max=cfg['data_type_max'])
                return probe_out_dim_metric
            
    def cal_attn_calib_prune_metric(self, calib, weight, metric_type):
        calib = calib.to(weight.device)
        calib = calib.to(torch.float32)
        sorted_calib, _ = torch.sort(calib)
        # print('sorted_calib', sorted_calib)
        weight = weight.to(torch.float32)
        if 'ppwandaspsum' in metric_type:
            calib = torch.sum(calib, dim=0)
            probe_out_dim_metric = torch.sum((calib.reshape((1,-1)) * torch.pow(weight, 2)).reshape(self.model_config.hidden_size, self.model_config.num_attention_heads, -1), dim=(0, 2))
        elif 'ppwandasp' in metric_type:
            calib = torch.sum(calib, dim=0)
            probe_out_dim_metric = torch.linalg.vector_norm((calib.reshape((1,-1)) * torch.pow(weight, 2)).reshape(self.model_config.hidden_size, self.model_config.num_attention_heads, -1), ord=2, dim=(0, 2))
        elif 'wandasp' in metric_type:
            calib = torch.sum(calib, dim=0)
            probe_out_dim_metric = (torch.sqrt(calib).reshape((1,-1)) * torch.abs(weight)).sum(dim=0)
        elif 'flap' in metric_type:
            calib = torch.sum(calib, dim=0)
            probe_out_dim_metric = (calib * torch.sum(torch.pow(weight, 2), dim=0))
        
        sorted_values, sorted_indices = torch.sort(probe_out_dim_metric)

        # Print sorted tensor
        # print("Sorted tensor:", sorted_values)
        return probe_out_dim_metric

    def cal_mlp_calib_prune_metric(self, calib, weight, metric_type):
        calib = calib.to(weight.device)
        calib = calib.to(torch.float32)

        sorted_calib, _ = torch.sort(calib)
        # print('sorted_calib', sorted_calib)
        weight = weight.to(torch.float32)
        # if 'ppwandasp' in metric_type:
        #     calib = torch.sum(calib, dim=0).clamp(max=cfg['data_type_max'])
        #     probe_out_dim_metric = torch.linalg.vector_norm((calib.reshape((1,-1)) * torch.pow(weight, 2)), ord=2, dim=0).clamp(max=cfg['data_type_max'])
        # elif 'wandasp' in metric_type:
        #     calib = torch.sum(calib, dim=0).clamp(max=cfg['data_type_max'])
        #     probe_out_dim_metric = (torch.sqrt(calib).reshape((1,-1)) * torch.abs(weight)).sum(dim=0).clamp(max=cfg['data_type_max'])
        # elif 'flap' in metric_type:
        #     calib = torch.sum(calib, dim=0).clamp(max=cfg['data_type_max'])
        #     probe_out_dim_metric = (calib * torch.sum(torch.pow(weight, 2), dim=0)).clamp(max=cfg['data_type_max'])
        if 'ppwandaspsum' in metric_type:
            calib = torch.sum(calib, dim=0)
            probe_out_dim_metric = torch.sum((calib.reshape((1,-1)) * torch.pow(weight, 2)), dim=0)
        elif 'ppwandasp' in metric_type:
            calib = torch.sum(calib, dim=0)
            probe_out_dim_metric = torch.linalg.vector_norm((calib.reshape((1,-1)) * torch.pow(weight, 2)), ord=2, dim=0)
        elif 'wandasp' in metric_type:
            calib = torch.sum(calib, dim=0)
            probe_out_dim_metric = (torch.sqrt(calib).reshape((1,-1)) * torch.abs(weight)).sum(dim=0)
        elif 'flap' in metric_type:
            calib = torch.sum(calib, dim=0)
            probe_out_dim_metric = (calib * torch.sum(torch.pow(weight, 2), dim=0))
        
        sorted_values, sorted_indices = torch.sort(probe_out_dim_metric)

        # Print sorted tensor
        # print("Sorted tensor:", sorted_values)
        return probe_out_dim_metric
    
    def sort_attn_metric(self, probe_out_dim_metric, num_heads, head_dim, prune_way, prune_module, multiple, pruning_ratio=None):
        if prune_way == None:
            return None, None, num_heads

        prune_ratio = pruning_ratio if pruning_ratio is not None else self.prune_ratio
        if 'whole' in prune_way:    
            probe_out_dim_metric = probe_out_dim_metric.reshape(num_heads, -1)
            summed_metrics = torch.clamp(probe_out_dim_metric.sum(dim=-1), max=cfg['data_type_max'])
            sorted_value, sorted_indices = torch.sort(summed_metrics, dim=0)
            num_prune_heads = int(prune_ratio * num_heads)
            heads_to_preserve = sorted_indices[num_prune_heads:]
            print('heads_to_preserve', heads_to_preserve)
            heads_to_preserve = torch.sort(heads_to_preserve)[0]
            print('heads_to_preserve sort', heads_to_preserve)
            # sorted_heads, _ = torch.sort(heads_to_preserve)

            # print(sorted_heads)  # Output: tensor([1, 2, 3, 5, 8])
            full_indices_to_preserve = (torch.arange(head_dim, device=probe_out_dim_metric.device) + heads_to_preserve.unsqueeze(1) * head_dim).view(-1)
            num_heads = num_heads - num_prune_heads
            return full_indices_to_preserve, num_heads, heads_to_preserve
        else:
            raise NotImplementedError

    def sort_mlp_metric(self, probe_out_dim_metric, multiple, pruning_ratio=None):        
        prune_ratio = pruning_ratio if pruning_ratio is not None else self.prune_ratio
        sorted_value, sorted_indices = torch.sort(probe_out_dim_metric, dim=0)
        num_prune = int(prune_ratio * probe_out_dim_metric.shape[0])
        num_prune = nearest_multiple(num_prune, probe_out_dim_metric.shape[0], multiple)
        # return torch.sort(sorted_indices[num_prune:])[0], torch.sort(sorted_indices[:num_prune])[0]
        return sorted_indices[num_prune:], sorted_indices[:num_prune]
    
    def cal_multiple_flops_for_flap(self, model):
        if 'csr' in cfg['task_name']:
            approx_seq_len = 100
        elif 'clm' in cfg['task_name']:
            approx_seq_len = cfg['max_seq_len']

        if 'llama' in cfg['model_name']:
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            attn_flops = approx_seq_len * model.config.hidden_size * head_dim * 4 + approx_seq_len ** 2 * head_dim + approx_seq_len ** 2 * head_dim
            print('approx_seq_len * model.config.hidden_size * head_dim * 4 ', approx_seq_len * model.config.hidden_size * head_dim * 4 )
            print('attnflops', attn_flops)
            mlp_flops = approx_seq_len * model.config.hidden_size * 3 
            print('mlp_flops', mlp_flops)
            multiples = attn_flops / mlp_flops
            # GQA
            if ('llama-3' in cfg['model_name'] or 'llama-2-70b' in cfg['model_name']) and ('q_proj' in cfg['cust_tgt_modules'] or 'k_proj' in cfg['cust_tgt_modules'] or 'v_proj' in cfg['cust_tgt_modules'] or 'o_proj' in cfg['cust_tgt_modules']) :
                head_dim = model.config.hidden_size // model.config.num_attention_heads
                attn_flops = approx_seq_len * model.config.hidden_size * head_dim * 2 + approx_seq_len ** 2 * head_dim + approx_seq_len ** 2 * head_dim
                print('approx_seq_len * model.config.hidden_size * head_dim * 2 ', approx_seq_len * model.config.hidden_size * head_dim * 2 )
                print('attnflops', attn_flops)
                mlp_flops = approx_seq_len * model.config.hidden_size * 3 
                print('mlp_flops', mlp_flops)
                multiples = attn_flops / mlp_flops
        elif 'opt' in cfg['model_name']:
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            attn_flops = approx_seq_len * model.config.hidden_size * head_dim * 4 + approx_seq_len ** 2 * head_dim + approx_seq_len ** 2 * head_dim
            mlp_flops = approx_seq_len * model.config.hidden_size * 2 
            multiples = attn_flops / mlp_flops
        
        return multiples




    def flap_ratio(self, model, logger=None):
        
        prune_ratio = self.adjust_prune_ratio(cfg['prune_ratio'], model.config)
        attn_metric_list, mlp_metric_list = [], []
        standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

        if 'llama' in cfg['model_name']:
            for name, module in model.named_modules():   
                if 'down_proj' not in name and 'o_proj' not in name:
                    continue

                if 'down_proj' in name and 'down_proj' not in cfg['cust_tgt_modules']:
                    continue
                elif 'o_proj' in name and 'o_proj' not in cfg['cust_tgt_modules']:
                    continue

                # numbers = int(''.join(filter(str.isdigit, name)))
                # if numbers <= cfg['skip_layers']:
                #     continue

                if check_skip_layers(name):
                    continue

                if 'o_proj' in name:
                    # flap code: W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                    # we dont put the manually added square (only added for attn) here since it is unreasonable
                    metric = self.cal_attn_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric'])
                    if 'square' in cfg['prune_method']:
                        metric = metric ** 2
                    attn_metric_list.append(metric.to('cpu'))
                elif 'down_proj' in name:
                    metric = self.cal_mlp_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric'])
                    mlp_metric_list.append(metric.to('cpu'))
                
            if len(attn_metric_list) > 0:
                attn_metric = torch.stack(attn_metric_list).to(torch.float64)
                attn_metric = standarlization(attn_metric)
                print('attn_metric', attn_metric.shape)
                attn_metric = attn_metric.reshape(attn_metric.shape[0], -1, 128).mean(dim=2)
            else:
                attn_metric = None

            if len(mlp_metric_list) > 0:
                mlp_metric = torch.stack(mlp_metric_list).to(torch.float64)
                mlp_metric = standarlization(mlp_metric)
            else:
                mlp_metric = None

            # prune 1 head will lead to multiples times more flops pruned than 1 mlp channel
            multiples = self.cal_multiple_flops_for_flap(model)
            print('multiples', multiples, flush=True)
            if attn_metric is not None and mlp_metric is not None:
                prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
                flops_measurement = torch.cat([torch.full_like(attn_metric, multiples).view(-1), torch.full_like(mlp_metric, 1).view(-1)])
            elif attn_metric is not None:
                prune_metric = attn_metric.view(-1)
                flops_measurement = torch.full_like(attn_metric, multiples).view(-1)
            elif mlp_metric is not None:
                prune_metric = mlp_metric.view(-1)
                flops_measurement = torch.full_like(mlp_metric, 1).view(-1)

            # descending=True
            sorted_prune, indices = torch.sort(prune_metric)
            sorted_flops_measurement = flops_measurement[indices]
            # threshold = sorted_prune[int(sorted_prune.numel() * cfg['prune_ratio'])]
            cumulative_sum = torch.cumsum(sorted_flops_measurement, dim=0)

            # Calculate the total sum and the target sum
            total_sum = sorted_flops_measurement.sum()
            target_sum = total_sum * prune_ratio
            
            # Find the index where the cumulative sum reaches or exceeds the target sum
            threshold_index = torch.searchsorted(cumulative_sum, target_sum, right=True)
            print("Total sum:", total_sum, target_sum, prune_ratio, threshold_index)
            # Set the threshold using the value at this index
            threshold = sorted_prune[threshold_index]

            print("Threshold:", threshold.item())

            attention_ratio = 0
            attention_counter = 0
            mlp_ratio = 0
            mlp_counter = 0
            for name, module in model.named_modules():
                if 'down_proj' not in name and 'o_proj' not in name:
                    continue
                
                if 'down_proj' in name and 'down_proj' not in cfg['cust_tgt_modules']:
                    continue
                elif 'o_proj' in name and 'o_proj' not in cfg['cust_tgt_modules']:
                    continue

                if check_skip_layers(name):
                    continue
                
                if 'o_proj' in name:
                    # flap code: W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                    # we dont put the manually added square (only added for attn) here since it is unreasonable
                    metric = self.cal_attn_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric']).reshape(1, -1)
                    if 'square' in cfg['prune_method']:
                        metric = metric ** 2
                    metric = standarlization(metric)
                    metric = metric.reshape(-1, 128).mean(dim=1)
                elif 'down_proj' in name:
                    metric = self.cal_mlp_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric']).reshape(1, -1)
                    metric = standarlization(metric)

                threshold = threshold.to(metric.device)
                module.pruning_ratio = metric[metric <= threshold].numel() / metric.numel()
                if 'o_proj' in name:
                    attention_ratio += module.pruning_ratio
                    attention_counter += 1
                elif 'down_proj' in name:
                    mlp_ratio += module.pruning_ratio
                    mlp_counter += 1
                print('name', name, 'module.pruning_ratio', module.pruning_ratio)
            
            logger.append({f'flap_attention_average_pruning_ratio': attention_ratio/(attention_counter+1e-4)}, 'test')
            logger.append({f'flap_mlp_average_pruning_ratio': mlp_ratio/(mlp_counter+1e-4)}, 'test')
        elif 'opt' in cfg['model_name']:
            for name, module in model.named_modules():   
                if 'fc2' not in name and 'out_proj' not in name:
                    continue

                if 'fc2' in name and 'fc2' not in cfg['cust_tgt_modules']:
                    continue
                elif 'out_proj' in name and 'out_proj' not in cfg['cust_tgt_modules']:
                    continue

                numbers = int(''.join(filter(str.isdigit, name)))
                # if numbers <= cfg['skip_layers']:
                #     continue

                if check_skip_layers(name):
                    continue

                if 'out_proj' in name:
                    # flap code: W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                    # we dont put the manually added square (only added for attn) here since it is unreasonable
                    metric = self.cal_attn_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric'])
                    if 'square' in cfg['prune_method']:
                        metric = metric ** 2
                    attn_metric_list.append(metric.to('cpu'))
                elif 'fc2' in name:
                    metric = self.cal_mlp_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric'])
                    mlp_metric_list.append(metric.to('cpu'))
                
            if len(attn_metric_list) > 0:
                attn_metric = torch.stack(attn_metric_list).to(torch.float64)
                attn_metric = standarlization(attn_metric)
                attn_metric = attn_metric.reshape(attn_metric.shape[0], -1, 128).mean(dim=2)
            else:
                attn_metric = None

            if len(mlp_metric_list) > 0:
                mlp_metric = torch.stack(mlp_metric_list).to(torch.float64)
                mlp_metric = standarlization(mlp_metric)
            else:
                mlp_metric = None

            # prune 1 head will lead to multiples times more flops pruned than 1 mlp channel
            multiples = self.cal_multiple_flops_for_flap(model)
            print('multiples', multiples, flush=True)
            if attn_metric is not None and mlp_metric is not None:
                prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
                flops_measurement = torch.cat([torch.full_like(attn_metric, multiples).view(-1), torch.full_like(mlp_metric, 1).view(-1)])
            elif attn_metric is not None:
                prune_metric = attn_metric.view(-1)
                flops_measurement = torch.full_like(attn_metric, multiples).view(-1)
            elif mlp_metric is not None:
                prune_metric = mlp_metric.view(-1)
                flops_measurement = torch.full_like(mlp_metric, 1).view(-1)

            # descending=True
            sorted_prune, indices = torch.sort(prune_metric)
            sorted_flops_measurement = flops_measurement[indices]
            # threshold = sorted_prune[int(sorted_prune.numel() * cfg['prune_ratio'])]
            cumulative_sum = torch.cumsum(sorted_flops_measurement, dim=0)

            # Calculate the total sum and the target sum
            total_sum = sorted_flops_measurement.sum()
            target_sum = total_sum * prune_ratio
            
            # Find the index where the cumulative sum reaches or exceeds the target sum
            threshold_index = torch.searchsorted(cumulative_sum, target_sum, right=True)
            print("Total sum:", total_sum, target_sum, prune_ratio, threshold_index)
            # Set the threshold using the value at this index
            threshold = sorted_prune[threshold_index]

            print("Threshold:", threshold.item())

            for name, module in model.named_modules():
                if 'fc2' not in name and 'out_proj' not in name:
                    continue
                
                if 'fc2' in name and 'fc2' not in cfg['cust_tgt_modules']:
                    continue
                elif 'out_proj' in name and 'out_proj' not in cfg['cust_tgt_modules']:
                    continue

                if check_skip_layers(name):
                    continue
                
                if 'out_proj' in name:
                    # flap code: W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                    # we dont put the manually added square (only added for attn) here since it is unreasonable
                    metric = self.cal_attn_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric']).reshape(1, -1)
                    if 'square' in cfg['prune_method']:
                        metric = metric ** 2
                    metric = standarlization(metric)
                    metric = metric.reshape(-1, 128).mean(dim=1)
                elif 'fc2' in name:
                    metric = self.cal_mlp_calib_prune_metric(module.get_global_metric_score_distribution(), module.weight.data, cfg['prune_metric']).reshape(1, -1)
                    metric = standarlization(metric)

                threshold = threshold.to(metric.device)
                module.pruning_ratio = metric[metric <= threshold].numel() / metric.numel()
                print('name', name, 'module.pruning_ratio', module.pruning_ratio)
        return

    # def grid_ratio(self, model):
    #     from module import TRANSFORMERS_MODELS_TO_GRID_SEARCH_RATIO
    #     if 'llama' in cfg['model_name']:
    #         for name, module in model.named_modules():
    #             if 'down_proj' not in name and 'o_proj' not in name:
    #                 continue
                
    #             if 'down_proj' in name and 'down_proj' not in cfg['cust_tgt_modules']:
    #                 continue
    #             elif 'o_proj' in name and 'o_proj' not in cfg['cust_tgt_modules']:
    #                 continue

    #             numbers = int(''.join(filter(str.isdigit, name)))
    #             print('numebres', numbers)
    #             if numbers <= cfg['skip_layers']:
    #                 continue
                
    #             if 'o_proj' in name:
    #                 # if there are more layers, like llama-2-13B, prune less in each layer to reach mean prune ratio
    #                 # simply scale this ratio, one may run the grid search for other model type to find the best ratio
    #                 pruning_ratio = TRANSFORMERS_MODELS_TO_GRID_SEARCH_RATIO['llama']['128'][cfg['prune_ratio']]['o_proj'] * \
    #                     (32 / 29) / (model.config.num_hidden_layers / (model.config.num_hidden_layers - (cfg['skip_layers'] + 1)))
    #                 module.pruning_ratio = pruning_ratio
    #             elif 'down_proj' in name:
    #                 pruning_ratio = TRANSFORMERS_MODELS_TO_GRID_SEARCH_RATIO['llama']['128'][cfg['prune_ratio']]['down_proj'] * \
    #                     (32 / 29) / (model.config.num_hidden_layers / (model.config.num_hidden_layers - (cfg['skip_layers'] + 1)))
    #                 module.pruning_ratio = pruning_ratio


    #     elif 'opt' in cfg['model_name']:
    #         pass
    #     return


    # def grid_search(self, model):
    #     if 'llama' in cfg['model_name']:
    #         for name, module in model.named_modules():
    #             if 'down_proj' not in name and 'o_proj' not in name:
    #                 continue
                
    #             if 'down_proj' in name and 'down_proj' not in cfg['cust_tgt_modules']:
    #                 continue
    #             elif 'o_proj' in name and 'o_proj' not in cfg['cust_tgt_modules']:
    #                 continue

    #             numbers = int(''.join(filter(str.isdigit, name)))
    #             print('numebers', numbers)
    #             if numbers <= cfg['skip_layers']:
    #                 continue
                
    #             if 'o_proj' in name:
    #                 module.pruning_ratio = cfg['prune_ratio'][0]
    #             elif 'down_proj' in name:
    #                 module.pruning_ratio = cfg['prune_ratio'][1]


    #     elif 'opt' in cfg['model_name']:
    #         pass
    #     return