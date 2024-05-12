#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
CUDA_VISIBLE_DEVICES=1 python test_local_tuned_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-2-7b_clm_20_1024_0.2_None_llmpruner-tune_asyncinter_None_None_None -- &> wslout/output_wikitext-2v1_llama-2-7b_clm_20_1024_0.2_None_llmpruner-tune_asyncinter_None_None_None_$timestamp.txt
wait
