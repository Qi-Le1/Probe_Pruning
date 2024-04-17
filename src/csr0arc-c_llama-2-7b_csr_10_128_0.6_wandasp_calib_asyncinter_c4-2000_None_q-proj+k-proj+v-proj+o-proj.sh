#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name arc-c_llama-2-7b_csr_10_128_0.6_wandasp_calib_asyncinter_c4-2000_None_q-proj+k-proj+v-proj+o-proj &> wslout/output_arc-c_llama-2-7b_csr_10_128_0.6_wandasp_calib_asyncinter_c4-2000_None_q-proj+k-proj+v-proj+o-proj_$timestamp.txt
wait
