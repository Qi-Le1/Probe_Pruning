#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_dense_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name boolq_llama-2-7b_clm_20_2048_0_None_dense_None_None_None_None &> wslout/output_boolq_llama-2-7b_clm_20_2048_0_None_dense_None_None_None_None_$timestamp.txt
wait
