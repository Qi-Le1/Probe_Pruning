#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-2-7b_clm_10_128_0.5_wandasp_calib-ema-probe_sync_c4-200_0.1normwhole4-0.1normwhole4-0.1normwhole4-None-None-seqrank_q-proj+k-proj+v-proj+o-proj &> wslout/output_wikitext-2v1_llama-2-7b_clm_10_128_0.5_wandasp_calib-ema-probe_sync_c4-200_0.1normwhole4-0.1normwhole4-0.1normwhole4-None-None-seqrank_q-proj+k-proj+v-proj+o-proj_$timestamp.txt
wait
