#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-3-8b_clm_1_128_0.4_flap_flap-default_asyncinter_c4-20_None_default &> wslout/output_wikitext-2v1_llama-3-8b_clm_1_128_0.4_flap_flap-default_asyncinter_c4-20_None_default_$timestamp.txt
wait
