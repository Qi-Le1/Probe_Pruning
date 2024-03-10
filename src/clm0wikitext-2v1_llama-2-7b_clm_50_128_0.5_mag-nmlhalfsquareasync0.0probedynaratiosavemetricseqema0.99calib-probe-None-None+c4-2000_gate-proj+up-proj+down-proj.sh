#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-2-7b_clm_50_128_0.5_mag-nmlhalfsquareasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000_gate-proj+up-proj+down-proj &> wslout/output_wikitext-2v1_llama-2-7b_clm_50_128_0.5_mag-nmlhalfsquareasync0.0probedynaratiosavemetricseqema0.99calib-probe-None-None+c4-2000_gate-proj+up-proj+down-proj_$timestamp.txt
wait
