#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_dense_model.py --device cuda --resume_mode 0 --init_seed 2 --control_name hellaswag_llama-2-7b_csr_20_512_0_None_dense_None_None_None_None &> wslout/output_hellaswag_llama-2-7b_csr_20_512_0_None_dense_None_None_None_None_$timestamp.txt
wait
