import argparse
import itertools
import os
import errno

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--pbs_prefix', default=None, type=str)
parser.add_argument('--code_folder', default=None, type=str)
parser.add_argument('--res_folder', default=None, type=str)
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--file', default=None, type=str)
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--log_interval', default=None, type=float)
args = vars(parser.parse_args())


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


# first run dense model, get dense running time and flops
# second run 1 2000c4 calib for flap/wandasp/ppwandasp for each model to store calibration info
# then run all exps

def main():
    controls = [
        # 'python hf_prune.py --seed 0 --base_model llama2-13b --pruning_ratio 0.22 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 40 --block_attention_layer_start 3 --block_attention_layer_end 40 --save_ckpt_log_name 0_llmpruner_llama-2-13b_0.2 --save_model',
        # 'python hf_prune.py --seed 0 --base_model llama2-13b --pruning_ratio 0.44 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 40 --block_attention_layer_start 3 --block_attention_layer_end 40 --save_ckpt_log_name 0_llmpruner_llama-2-13b_0.4 --save_model',
        # 'python hf_prune.py --seed 0 --base_model llama2-13b --pruning_ratio 0.66 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 40 --block_attention_layer_start 3 --block_attention_layer_end 40 --save_ckpt_log_name 0_llmpruner_llama-2-13b_0.6 --save_model',
        # 'python hf_prune.py --seed 1 --base_model llama2-13b --pruning_ratio 0.22 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 40 --block_attention_layer_start 3 --block_attention_layer_end 40 --save_ckpt_log_name 1_llmpruner_llama-2-13b_0.2 --save_model',
        # 'python hf_prune.py --seed 1 --base_model llama2-13b --pruning_ratio 0.44 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 40 --block_attention_layer_start 3 --block_attention_layer_end 40 --save_ckpt_log_name 1_llmpruner_llama-2-13b_0.4 --save_model',
        # 'python hf_prune.py --seed 1 --base_model llama2-13b --pruning_ratio 0.66 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 40 --block_attention_layer_start 3 --block_attention_layer_end 40 --save_ckpt_log_name 1_llmpruner_llama-2-13b_0.6 --save_model',
        # 'python hf_prune.py --seed 2 --base_model llama2-13b --pruning_ratio 0.22 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 40 --block_attention_layer_start 3 --block_attention_layer_end 40 --save_ckpt_log_name 2_llmpruner_llama-2-13b_0.2 --save_model',
        # 'python hf_prune.py --seed 2 --base_model llama2-13b --pruning_ratio 0.44 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 40 --block_attention_layer_start 3 --block_attention_layer_end 40 --save_ckpt_log_name 2_llmpruner_llama-2-13b_0.4 --save_model',
        # 'python hf_prune.py --seed 2 --base_model llama2-13b --pruning_ratio 0.66 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 40 --block_attention_layer_start 3 --block_attention_layer_end 40 --save_ckpt_log_name 2_llmpruner_llama-2-13b_0.6 --save_model',
        # 'python hf_prune.py --seed 0 --base_model llama2-7b --pruning_ratio 0.22 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 32 --block_attention_layer_start 3 --block_attention_layer_end 32 --save_ckpt_log_name 0_llmpruner_llama-2-7b_0.2 --save_model',
        # 'python hf_prune.py --seed 0 --base_model llama2-7b --pruning_ratio 0.44 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 32 --block_attention_layer_start 3 --block_attention_layer_end 32 --save_ckpt_log_name 0_llmpruner_llama-2-7b_0.4 --save_model',
        # 'python hf_prune.py --seed 0 --base_model llama2-7b --pruning_ratio 0.66 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 32 --block_attention_layer_start 3 --block_attention_layer_end 32 --save_ckpt_log_name 0_llmpruner_llama-2-7b_0.6 --save_model',
        # 'python hf_prune.py --seed 1 --base_model llama2-7b --pruning_ratio 0.22 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 32 --block_attention_layer_start 3 --block_attention_layer_end 32 --save_ckpt_log_name 1_llmpruner_llama-2-7b_0.2 --save_model',
        # 'python hf_prune.py --seed 1 --base_model llama2-7b --pruning_ratio 0.44 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 32 --block_attention_layer_start 3 --block_attention_layer_end 32 --save_ckpt_log_name 1_llmpruner_llama-2-7b_0.4 --save_model',
        # 'python hf_prune.py --seed 1 --base_model llama2-7b --pruning_ratio 0.66 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 32 --block_attention_layer_start 3 --block_attention_layer_end 32 --save_ckpt_log_name 1_llmpruner_llama-2-7b_0.6 --save_model',
        # 'python hf_prune.py --seed 2 --base_model llama2-7b --pruning_ratio 0.22 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 32 --block_attention_layer_start 3 --block_attention_layer_end 32 --save_ckpt_log_name 2_llmpruner_llama-2-7b_0.2 --save_model',
        # 'python hf_prune.py --seed 2 --base_model llama2-7b --pruning_ratio 0.44 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 32 --block_attention_layer_start 3 --block_attention_layer_end 32 --save_ckpt_log_name 2_llmpruner_llama-2-7b_0.4 --save_model',
        # 'python hf_prune.py --seed 2 --base_model llama2-7b --pruning_ratio 0.66 --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 32 --block_attention_layer_start 3 --block_attention_layer_end 32 --save_ckpt_log_name 2_llmpruner_llama-2-7b_0.6 --save_model',

        # 'python prune.py --seed 0 --base_model llama2-13b --ratio 0.22 --init_ratio 0.22 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/0_loraprune_llama-2-13b_0.2 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 0 --base_model llama2-13b --ratio 0.44 --init_ratio 0.44 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/0_loraprune_llama-2-13b_0.4 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 0 --base_model llama2-13b --ratio 0.66 --init_ratio 0.66 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/0_loraprune_llama-2-13b_0.6 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 1 --base_model llama2-13b --ratio 0.22 --init_ratio 0.22 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/1_loraprune_llama-2-13b_0.2 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 1 --base_model llama2-13b --ratio 0.44 --init_ratio 0.44 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/1_loraprune_llama-2-13b_0.4 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 1 --base_model llama2-13b --ratio 0.66 --init_ratio 0.66 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/1_loraprune_llama-2-13b_0.6 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 2 --base_model llama2-13b --ratio 0.22 --init_ratio 0.22 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/2_loraprune_llama-2-13b_0.2 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 2 --base_model llama2-13b --ratio 0.66 --init_ratio 0.66 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/2_loraprune_llama-2-13b_0.6 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 2 --base_model llama2-13b --ratio 0.44 --init_ratio 0.44 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/2_loraprune_llama-2-13b_0.4 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 0 --base_model llama2-7b --ratio 0.22 --init_ratio 0.22 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/0_loraprune_llama-2-7b_0.2 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 0 --base_model llama2-7b --ratio 0.44 --init_ratio 0.44 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/0_loraprune_llama-2-7b_0.4 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 1 --base_model llama2-7b --ratio 0.22 --init_ratio 0.22 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/1_loraprune_llama-2-7b_0.2 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 0 --base_model llama2-7b --ratio 0.66 --init_ratio 0.66 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/0_loraprune_llama-2-7b_0.6 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 1 --base_model llama2-7b --ratio 0.44 --init_ratio 0.44 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/1_loraprune_llama-2-7b_0.4 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 1 --base_model llama2-7b --ratio 0.66 --init_ratio 0.66 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/1_loraprune_llama-2-7b_0.6 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 2 --base_model llama2-7b --ratio 0.66 --init_ratio 0.66 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/2_loraprune_llama-2-7b_0.6 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 2 --base_model llama2-7b --ratio 0.22 --init_ratio 0.22 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/2_loraprune_llama-2-7b_0.2 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',
        # 'python prune.py --seed 2 --base_model llama2-7b --ratio 0.44 --init_ratio 0.44 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/2_loraprune_llama-2-7b_0.4 --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1',

        'python post_training.py --seed 0 --prune_model prune_log/0_llmpruner_llama-2-13b_0.2/pytorch_model.bin --output_dir tune_log/0_llmpruner_llama-2-13b_0.2 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 0 --prune_model prune_log/0_llmpruner_llama-2-13b_0.4/pytorch_model.bin --output_dir tune_log/0_llmpruner_llama-2-13b_0.4 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 0 --prune_model prune_log/0_llmpruner_llama-2-13b_0.6/pytorch_model.bin --output_dir tune_log/0_llmpruner_llama-2-13b_0.6 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 1 --prune_model prune_log/1_llmpruner_llama-2-13b_0.2/pytorch_model.bin --output_dir tune_log/1_llmpruner_llama-2-13b_0.2 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 1 --prune_model prune_log/1_llmpruner_llama-2-13b_0.4/pytorch_model.bin --output_dir tune_log/1_llmpruner_llama-2-13b_0.4 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 1 --prune_model prune_log/1_llmpruner_llama-2-13b_0.6/pytorch_model.bin --output_dir tune_log/1_llmpruner_llama-2-13b_0.6 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 2 --prune_model prune_log/2_llmpruner_llama-2-13b_0.2/pytorch_model.bin --output_dir tune_log/2_llmpruner_llama-2-13b_0.2 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 2 --prune_model prune_log/2_llmpruner_llama-2-13b_0.6/pytorch_model.bin --output_dir tune_log/2_llmpruner_llama-2-13b_0.6 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 2 --prune_model prune_log/2_llmpruner_llama-2-13b_0.4/pytorch_model.bin --output_dir tune_log/2_llmpruner_llama-2-13b_0.4 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 0 --prune_model prune_log/0_llmpruner_llama-2-7b_0.2/pytorch_model.bin --output_dir tune_log/0_llmpruner_llama-2-7b_0.2 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 0 --prune_model prune_log/0_llmpruner_llama-2-7b_0.4/pytorch_model.bin --output_dir tune_log/0_llmpruner_llama-2-7b_0.4 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 0 --prune_model prune_log/0_llmpruner_llama-2-7b_0.6/pytorch_model.bin --output_dir tune_log/0_llmpruner_llama-2-7b_0.6 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 1 --prune_model prune_log/1_llmpruner_llama-2-7b_0.2/pytorch_model.bin --output_dir tune_log/1_llmpruner_llama-2-7b_0.2 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 1 --prune_model prune_log/1_llmpruner_llama-2-7b_0.4/pytorch_model.bin --output_dir tune_log/1_llmpruner_llama-2-7b_0.4 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 1 --prune_model prune_log/1_llmpruner_llama-2-7b_0.6/pytorch_model.bin --output_dir tune_log/1_llmpruner_llama-2-7b_0.6 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 2 --prune_model prune_log/2_llmpruner_llama-2-7b_0.2/pytorch_model.bin --output_dir tune_log/2_llmpruner_llama-2-7b_0.2 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 2 --prune_model prune_log/2_llmpruner_llama-2-7b_0.6/pytorch_model.bin --output_dir tune_log/2_llmpruner_llama-2-7b_0.6 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python post_training.py --seed 2 --prune_model prune_log/2_llmpruner_llama-2-7b_0.4/pytorch_model.bin --output_dir tune_log/2_llmpruner_llama-2-7b_0.4 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs',
        'python prune.py --seed 0 --base_model llama2-13b --ratio 0.22 --output_dir outputs_dir/0_loraprune_llama-2-13b_0.2',
        'python prune.py --seed 0 --base_model llama2-13b --ratio 0.44 --output_dir outputs_dir/0_loraprune_llama-2-13b_0.4',
        'python prune.py --seed 0 --base_model llama2-13b --ratio 0.66 --output_dir outputs_dir/0_loraprune_llama-2-13b_0.6',
        'python prune.py --seed 1 --base_model llama2-13b --ratio 0.22 --output_dir outputs_dir/1_loraprune_llama-2-13b_0.2',
        'python prune.py --seed 1 --base_model llama2-13b --ratio 0.44 --output_dir outputs_dir/1_loraprune_llama-2-13b_0.4',
        'python prune.py --seed 1 --base_model llama2-13b --ratio 0.66 --output_dir outputs_dir/1_loraprune_llama-2-13b_0.6',
        'python prune.py --seed 2 --base_model llama2-13b --ratio 0.66 --output_dir outputs_dir/2_loraprune_llama-2-13b_0.6',
        'python prune.py --seed 2 --base_model llama2-13b --ratio 0.22 --output_dir outputs_dir/2_loraprune_llama-2-13b_0.2',
        'python prune.py --seed 2 --base_model llama2-13b --ratio 0.44 --output_dir outputs_dir/2_loraprune_llama-2-13b_0.4',
        'python prune.py --seed 0 --base_model llama2-7b --ratio 0.22 --output_dir outputs_dir/0_loraprune_llama-2-7b_0.2',
        'python prune.py --seed 0 --base_model llama2-7b --ratio 0.66 --output_dir outputs_dir/0_loraprune_llama-2-7b_0.6',
        'python prune.py --seed 0 --base_model llama2-7b --ratio 0.44 --output_dir outputs_dir/0_loraprune_llama-2-7b_0.4',
        'python prune.py --seed 1 --base_model llama2-7b --ratio 0.22 --output_dir outputs_dir/1_loraprune_llama-2-7b_0.2',
        'python prune.py --seed 1 --base_model llama2-7b --ratio 0.44 --output_dir outputs_dir/1_loraprune_llama-2-7b_0.4',
        'python prune.py --seed 1 --base_model llama2-7b --ratio 0.66 --output_dir outputs_dir/1_loraprune_llama-2-7b_0.6',
        'python prune.py --seed 2 --base_model llama2-7b --ratio 0.66 --output_dir outputs_dir/2_loraprune_llama-2-7b_0.6',
        'python prune.py --seed 2 --base_model llama2-7b --ratio 0.22 --output_dir outputs_dir/2_loraprune_llama-2-7b_0.2',
        'python prune.py --seed 2 --base_model llama2-7b --ratio 0.44 --output_dir outputs_dir/2_loraprune_llama-2-7b_0.4',

    ]



    # k = 0
    # s_for_max = '#!/bin/bash\n'
    # k_for_max = 0
    import os

    

    task_parallel_num = 1
    mem = 80

    i = 0
    while i < len(controls):
        print('%$%$$controls', controls)
        if 'loraprune' in controls[i]:
            code_folder = 'LoRAPruneCodeRelease'
        else:
            code_folder = 'llm_pruner_mod'

        # for i in range(len(controls)):
        # controls[i] = list(controls[i])
        print('controls[i]', controls[i])
        temp = controls[i]
        filename = ''.join(str(_) for _ in controls[i][:20])
        filename = filename + str(i)
        filename = code_folder + filename
        # print(f'filename: {filename}')
        if 'loraprune' in controls[i]:
            if 'warmup' in controls[i]:
                run_time = '01:00:00'
            else:
                run_time = '12:00:00'
        else:
            if 'hf_prune' in controls[i]:
                run_time = '01:00:00'
            else:
                run_time = '09:00:00'

        res_folder = 'msiout'
        res_path = os.path.join(f'/home/aanwar/le000288/{code_folder}/', res_folder)
        print(res_path)
        makedir_exist_ok(res_folder)

        bash_file_name = './{}.bash'.format(f'msi_{code_folder}')

        # def delete_file_if_exist(file_name):
        #     if os.path.exists(file_name):
        #         # Delete the file if it exists
        #         os.remove(file_name)

        # # Check if the file exists

        # delete_file_if_exist(bash_file_name)

        gpu_num = 1
        # if 'clm' in controls[i-1][4] and '2048' in controls[i-1][4]:
        #     if '30b' in controls[i-1][4] or '70b' in controls[i-1][4]:
        #     #     gpu_num = 3
        #     # else:
        #     #     gpu_num = 2
        #         gpu_num = 2
        #     else:
        #         gpu_num = 2

        # if 'probe' in controls[i-1][4]:
        #     if '30b' in controls[i-1][4] or '70b' in controls[i-1][4]:
        #         gpu_num = 2
        #     else:
        #         gpu_num = 1
        # else:
        #     if '30b' in controls[i-1][4] or '70b' in controls[i-1][4]:
        #         gpu_num = 3
        #     elif '13b' in controls[i-1][4]:
        #         gpu_num = 2
        #     else:
        #         gpu_num = 1
        s = '#!/bin/bash -l\n'

        s += f'#SBATCH --time={run_time}\n'
        s += f'#SBATCH --nodes={task_parallel_num}\n'
        s += f'#SBATCH --ntasks={task_parallel_num}\n'
        # s += '#SBATCH --cpus-per-task=2'
        # s += '#SBATCH --gres=gpu:a100:1\n'

        s += f'#SBATCH -A aanwar\n'
        s += f'#SBATCH --gres=gpu:a100:{gpu_num}\n'
        s += '#SBATCH --partition=a100-8\n'

        # s += f'#SBATCH -A dingj\n'
        # s += f'#SBATCH --gres=gpu:{gpu_num}\n'
        # s += '#SBATCH --partition=jd-4a100\n'

        s += f'#SBATCH --mem={mem}gb\n'
        # s += '#SBATCH --mail-type=ALL \n'
        # s += '#SBATCH --mail-user=le000288@umn.edu\n'
        s += f'#SBATCH -o {res_path}/%j_{filename}.out\n'
        s += f'#SBATCH -e {res_path}/%j_{filename}.err\n'
        s += '\n'



        s += f'cd /home/aanwar/le000288/{code_folder}/\n'
        s += '\n'
        s += 'export PATH=/home/aanwar/le000288/miniconda3/envs/localtunedllm/bin:$PATH\n'
        # if 'max' in controls[i][-1]:
        #     s_for_max = s_for_max + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
        #         '--resume_mode {} --log_interval {} --device {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])

        #     if k_for_max % round == round - 1:
        #         s_for_max = s_for_max[:-2] + '\nwait\n'
        #     k_for_max = k_for_max + 1
        #     continue
        # while i < len(controls):
        # srun --nodes=1 --ntasks=1
        # srun --nodes=1 --ntasks=1

        s += '\n'
        s+= controls[i]

        s += 'wait\n'
        # controls[i][0] = 'test_classifier_fl.py'
        # for item in sub_controls:
        #     item[0] = item[0].replace('train', 'test')
        #     print(item, item[0])
        #     s += '\n'
        #     s = s + 'srun --nodes=1 --ntasks=1 python {} --device {} --resume_mode {} --init_seed {} --control_name {}&\n'.format(*item)
        # s += 'wait\n'
        pbs_file_name = './{}.pbs'.format(f'{filename}')
        # Check if the file exists
        if os.path.exists(pbs_file_name):
            # Delete the file if it exists
            os.remove(pbs_file_name)
        run_file = open(pbs_file_name, 'a')
        run_file.write(s)
        run_file.close()

        run_file = open(bash_file_name, 'a')
        command = f'sbatch {filename}.pbs --wait\n'
        run_file.write(command)
        run_file.close()
        i += 1
        # with open(bash_file_name, 'r') as cur_file:
        #     line_count = sum(1 for line in cur_file)

        # if line_count > 180:
        #     bash_file_name = './{}.bash'.format(f'msi_{file}_{data[0]}_{i}')
        #     print('bash_file_name', bash_file_name)
        #     delete_file_if_exist(bash_file_name)
    return


if __name__ == '__main__':
    main()
