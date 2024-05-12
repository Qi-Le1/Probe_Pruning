#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-2-7b_clm_10_128_0.4_None_llmpruner-prune_None_None_None_default &> wslout/output_wikitext-2v1_llama-2-7b_clm_10_128_0.4_None_llmpruner-prune_None_None_None_default_$timestamp.txt
wait
