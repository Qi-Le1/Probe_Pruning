#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name arc-c_llama-2-7b_csr_10_128_0.5_wandasp_calib-ema-probe-norm_sync_c4-2000_0.1normwhole4-0.1normwhole4-0.1normwhole4-0.1norm4-0.1norm4-seqrank_default &> wslout/output_arc-c_llama-2-7b_csr_10_128_0.5_wandasp_calib-ema-probe-norm_sync_c4-2000_0.1normwhole4-0.1normwhole4-0.1normwhole4-0.1norm4-0.1norm4-seqrank_default_$timestamp.txt
wait
