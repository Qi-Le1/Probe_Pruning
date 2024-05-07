#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 2 --control_name obqa_llama-2-7b_csr_20_512_0.2_ppwandasp_probe-default_sync_c4-2000_0.1-0.1-0.1-0.1-0.1-bszrank_default &> wslout/output_obqa_llama-2-7b_csr_20_512_0.2_ppwandasp_probe-default_sync_c4-2000_0.1-0.1-0.1-0.1-0.1-bszrank_default_$timestamp.txt
wait
