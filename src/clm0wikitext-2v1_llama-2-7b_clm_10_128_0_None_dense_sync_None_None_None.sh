#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
nsys profile -w true --gpu-metrics-device=0 -x true --force-overwrite=true -o wikitext-2v1_llama-2-7b_clm_10_128_0_None_dense_sync_None_None_None python test_dense_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-2-7b_clm_10_128_0_None_dense_sync_None_None_None &> wslout/output_wikitext-2v1_llama-2-7b_clm_10_128_0_None_dense_sync_None_None_None_$timestamp.txt
wait
