#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-2-7b_clm_3_128_0.3_probe_calib-ema_sync_c4-15_None_gate-proj+up-proj+down-proj &> wslout/output_wikitext-2v1_llama-2-7b_clm_3_128_0.3_probe_calib-ema_sync_c4-15_None_gate-proj+up-proj+down-proj_$timestamp.txt
wait
