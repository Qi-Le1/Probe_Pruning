#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 2 --control_name piqa_llama-3-8b_csr_20_512_0.2_wandasp_wandasp-default_asyncinter_c4-2000_None_default &> wslout/output_piqa_llama-3-8b_csr_20_512_0.2_wandasp_wandasp-default_asyncinter_c4-2000_None_default_$timestamp.txt
wait
