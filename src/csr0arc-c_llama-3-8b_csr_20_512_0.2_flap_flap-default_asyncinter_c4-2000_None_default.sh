#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name arc-c_llama-3-8b_csr_20_512_0.2_flap_flap-default_asyncinter_c4-2000_None_default &> wslout/output_arc-c_llama-3-8b_csr_20_512_0.2_flap_flap-default_asyncinter_c4-2000_None_default_$timestamp.txt
wait
