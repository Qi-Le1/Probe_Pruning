#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-3-8b_clm_1_1024_0.6_flap_calib_asyncinter_c4-2000_None_gate-proj+up-proj+down-proj &> wslout/output_wikitext-2v1_llama-3-8b_clm_1_1024_0.6_flap_calib_asyncinter_c4-2000_None_gate-proj+up-proj+down-proj_$timestamp.txt
wait
