#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
CUDA_VISIBLE_DEVICES=0 python test_local_tuned_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name boolq_llama-2-7b_csr_20_512_0.2_None_llmpruner-prune_asyncinter_None_None_None -- &> wslout/output_boolq_llama-2-7b_csr_20_512_0.2_None_llmpruner-prune_asyncinter_None_None_None_$timestamp.txt
wait
