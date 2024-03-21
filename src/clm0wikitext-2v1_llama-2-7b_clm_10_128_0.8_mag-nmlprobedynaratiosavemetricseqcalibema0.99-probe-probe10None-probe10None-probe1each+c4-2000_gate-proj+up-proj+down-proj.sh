#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-2-7b_clm_10_128_0.8_mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe1each+c4-2000_gate-proj+up-proj+down-proj &> wslout/output_wikitext-2v1_llama-2-7b_clm_10_128_0.8_mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe10None-probe10None-probe1each+c4-2000_gate-proj+up-proj+down-proj_$timestamp.txt
wait
