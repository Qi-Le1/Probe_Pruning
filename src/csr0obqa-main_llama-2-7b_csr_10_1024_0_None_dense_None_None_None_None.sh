#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_dense_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name obqa-main_llama-2-7b_csr_10_1024_0_None_dense_None_None_None_None &> wslout/output_obqa-main_llama-2-7b_csr_10_1024_0_None_dense_None_None_None_None_$timestamp.txt
wait
