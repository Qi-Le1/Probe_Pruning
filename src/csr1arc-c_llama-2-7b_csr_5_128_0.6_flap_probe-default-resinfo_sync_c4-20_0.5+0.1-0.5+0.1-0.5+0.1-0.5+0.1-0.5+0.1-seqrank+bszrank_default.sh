#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 1 --control_name arc-c_llama-2-7b_csr_5_128_0.6_flap_probe-default-resinfo_sync_c4-20_0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-seqrank+bszrank_default &> wslout/output_arc-c_llama-2-7b_csr_5_128_0.6_flap_probe-default-resinfo_sync_c4-20_0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-seqrank+bszrank_default_$timestamp.txt
wait
