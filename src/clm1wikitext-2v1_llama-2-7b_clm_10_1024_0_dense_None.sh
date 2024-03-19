#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_dense_model.py --device cuda --resume_mode 0 --init_seed 1 --control_name wikitext-2v1_llama-2-7b_clm_10_1024_0_dense_None &> wslout/output_wikitext-2v1_llama-2-7b_clm_10_1024_0_dense_None_$timestamp.txt
wait
