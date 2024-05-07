#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 2 --control_name hellaswag_llama-3-8b_csr_20_512_0.2_ppwandasp_probe-default_sync_c4-2000_0.1-0.1-0.1-0.1-0.1-bszrank_default &> wslout/output_hellaswag_llama-3-8b_csr_20_512_0.2_ppwandasp_probe-default_sync_c4-2000_0.1-0.1-0.1-0.1-0.1-bszrank_default_$timestamp.txt
wait
