#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name ptb_llama-3-8b_clm_20_1024_0.5_wandasp_wandasp-default_asyncinter_c4-2000_None_default &> wslout/output_ptb_llama-3-8b_clm_20_1024_0.5_wandasp_wandasp-default_asyncinter_c4-2000_None_default_$timestamp.txt
wait
