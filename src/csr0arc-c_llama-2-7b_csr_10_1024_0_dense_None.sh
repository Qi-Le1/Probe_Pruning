#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
nsys profile -w true --gpu-metrics-device=0 -x true --force-overwrite=true -o arc-c_llama-2-7b_csr_10_1024_0_dense_None python test_dense_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name arc-c_llama-2-7b_csr_10_1024_0_dense_None &> wslout/output_arc-c_llama-2-7b_csr_10_1024_0_dense_None_$timestamp.txt
wait
