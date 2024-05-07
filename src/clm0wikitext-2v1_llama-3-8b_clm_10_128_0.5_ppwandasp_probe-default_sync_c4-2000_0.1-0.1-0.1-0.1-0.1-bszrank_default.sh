#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-3-8b_clm_10_128_0.5_ppwandasp_probe-default_sync_c4-2000_0.1-0.1-0.1-0.1-0.1-bszrank_default &> wslout/output_wikitext-2v1_llama-3-8b_clm_10_128_0.5_ppwandasp_probe-default_sync_c4-2000_0.1-0.1-0.1-0.1-0.1-bszrank_default_$timestamp.txt
wait
