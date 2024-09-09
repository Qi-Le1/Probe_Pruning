#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name arc-c_llama-2-7b_csr_20_512_0.4_ppwandasp_probe-default_sync_c4-2000_0.05+0.5-0.05+0.5-0.05+0.5-0.05+0.5-0.05+0.5-bszrank+seqrank_default &> wslout/output_arc-c_llama-2-7b_csr_20_512_0.4_ppwandasp_probe-default_sync_c4-2000_0.05+0.5-0.05+0.5-0.05+0.5-0.05+0.5-0.05+0.5-bszrank+seqrank_default_$timestamp.txt
wait
