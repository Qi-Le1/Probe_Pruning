#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name obqa-main_llama-2-7b_csr_10_128_0.5_ppwandasp_probe-default_sync_c4-20_0.2-0.2-0.2-0.2-0.2-bszrank_default &> wslout/output_obqa-main_llama-2-7b_csr_10_128_0.5_ppwandasp_probe-default_sync_c4-20_0.2-0.2-0.2-0.2-0.2-bszrank_default_$timestamp.txt
wait
