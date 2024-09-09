#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_opt-6.7b_clm_20_512_0.4_ppwandasp_probe-default-rulerank-last_sync_c4-2000_0.05-0.05-0.05-0.05-0.05-seqrank_default &> wslout/output_wikitext-2v1_opt-6.7b_clm_20_512_0.4_ppwandasp_probe-default-rulerank-last_sync_c4-2000_0.05-0.05-0.05-0.05-0.05-seqrank_default_$timestamp.txt
wait
