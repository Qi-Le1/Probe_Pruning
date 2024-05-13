#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
CUDA_VISIBLE_DEVICES=5 python test_local_tuned_model.py --device cuda --resume_mode 0 --init_seed 2 --control_name arc-e_llama-2-7b_csr_20_512_0.6_None_loraprune-tune_asyncinter_None_None_None -- &> wslout/output_arc-e_llama-2-7b_csr_20_512_0.6_None_loraprune-tune_asyncinter_None_None_None_$timestamp.txt
wait
