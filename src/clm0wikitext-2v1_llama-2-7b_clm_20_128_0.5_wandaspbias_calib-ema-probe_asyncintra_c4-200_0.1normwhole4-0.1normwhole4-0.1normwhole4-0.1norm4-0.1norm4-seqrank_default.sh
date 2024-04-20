#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-2-7b_clm_20_128_0.5_wandaspbias_calib-ema-probe_asyncintra_c4-200_0.1normwhole4-0.1normwhole4-0.1normwhole4-0.1norm4-0.1norm4-seqrank_default &> wslout/output_wikitext-2v1_llama-2-7b_clm_20_128_0.5_wandaspbias_calib-ema-probe_asyncintra_c4-200_0.1normwhole4-0.1normwhole4-0.1normwhole4-0.1norm4-0.1norm4-seqrank_default_$timestamp.txt
wait
