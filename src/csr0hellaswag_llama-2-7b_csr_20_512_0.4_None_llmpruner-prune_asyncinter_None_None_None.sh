#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
CUDA_VISIBLE_DEVICES=6 python test_local_tuned_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name hellaswag_llama-2-7b_csr_20_512_0.4_None_llmpruner-prune_asyncinter_None_None_None -- &> wslout/output_hellaswag_llama-2-7b_csr_20_512_0.4_None_llmpruner-prune_asyncinter_None_None_None_$timestamp.txt
wait
