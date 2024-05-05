#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_opt-6.7b_clm_10_128_0.4_ppwandasp_probe_sync_c4-2000_0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-seqrank+bszrank_default &> wslout/output_wikitext-2v1_opt-6.7b_clm_10_128_0.4_ppwandasp_probe_sync_c4-2000_0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-seqrank+bszrank_default_$timestamp.txt
wait
