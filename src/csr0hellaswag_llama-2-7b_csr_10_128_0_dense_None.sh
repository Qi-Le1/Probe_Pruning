#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_dense_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name hellaswag_llama-2-7b_csr_10_128_0_dense_None &> wslout/output_hellaswag_llama-2-7b_csr_10_128_0_dense_None_$timestamp.txt
wait
