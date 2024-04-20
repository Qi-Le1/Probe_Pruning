#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-2-7b_clm_10_128_0.7_wandasp_calib-ema-probe-resinfo0.1_asyncintra_c4-2000_0.1normwhole1-0.1normwhole1-0.1normwhole1-0.1norm1-0.1norm1-bszrank_default &> wslout/output_wikitext-2v1_llama-2-7b_clm_10_128_0.7_wandasp_calib-ema-probe-resinfo0.1_asyncintra_c4-2000_0.1normwhole1-0.1normwhole1-0.1normwhole1-0.1norm1-0.1norm1-bszrank_default_$timestamp.txt
wait
