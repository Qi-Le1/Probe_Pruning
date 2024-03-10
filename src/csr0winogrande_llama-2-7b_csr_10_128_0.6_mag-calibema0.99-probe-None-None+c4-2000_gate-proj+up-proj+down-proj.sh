#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name winogrande_llama-2-7b_csr_10_128_0.6_mag-calibema0.99-probe-None-None+c4-2000_gate-proj+up-proj+down-proj &> wslout/output_winogrande_llama-2-7b_csr_10_128_0.6_mag-calibema0.99-probe-None-None+c4-2000_gate-proj+up-proj+down-proj_$timestamp.txt
wait
