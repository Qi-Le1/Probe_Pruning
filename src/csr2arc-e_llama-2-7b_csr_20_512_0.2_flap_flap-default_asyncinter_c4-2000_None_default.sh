#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 2 --control_name arc-e_llama-2-7b_csr_20_512_0.2_flap_flap-default_asyncinter_c4-2000_None_default &> wslout/output_arc-e_llama-2-7b_csr_20_512_0.2_flap_flap-default_asyncinter_c4-2000_None_default_$timestamp.txt
wait
