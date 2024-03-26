#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name arc-c_llama-2-7b_csr_10_128_0.3_mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe2None-probe2None-probe2each+c4-2000_q-proj+k-proj+v-proj+o-proj &> wslout/output_arc-c_llama-2-7b_csr_10_128_0.3_mag-nmlprobedynaratiosavemetricseqcalibema0.99-probe-probe2None-probe2None-probe2each+c4-2000_q-proj+k-proj+v-proj+o-proj_$timestamp.txt
wait
