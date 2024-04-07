#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
nsys profile -w true --gpu-metrics-device=0 -x true --force-overwrite=true -o wikitext-2v1_llama-2-7b_clm_100_448_0.0_probe_calib-probe-ema_asyncintra_c4-15_0-0-0-5-5_gate-proj+up-proj+down-proj python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-2-7b_clm_100_448_0.0_probe_calib-probe-ema_asyncintra_c4-15_0-0-0-5-5_gate-proj+up-proj+down-proj &> wslout/output_wikitext-2v1_llama-2-7b_clm_100_448_0.0_probe_calib-probe-ema_asyncintra_c4-15_0-0-0-5-5_gate-proj+up-proj+down-proj_$timestamp.txt
wait
