#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name arc-c+wikitext-2v1_llama-2-7b_mix_20_256_0.4_ppwandasp_probe-default_sync_c4-20_0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank_default &> wslout/output_arc-c+wikitext-2v1_llama-2-7b_mix_20_256_0.4_ppwandasp_probe-default_sync_c4-20_0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank_default_$timestamp.txt
wait
