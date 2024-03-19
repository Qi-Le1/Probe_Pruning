#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_opt-6.7b_clm_10_128_0.8_mag-nmlmultiprobe1probedynaratiosavemetricseqema0.99calib-probe-None-None-None+c4-2000_fc1+fc2 &> wslout/output_wikitext-2v1_opt-6.7b_clm_10_128_0.8_mag-nmlmultiprobe1probedynaratiosavemetricseqema0.99calib-probe-None-None-None+c4-2000_fc1+fc2_$timestamp.txt
wait
