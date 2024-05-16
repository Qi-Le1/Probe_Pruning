#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_opt-6.7b_clm_5_128_0.6_wandasp_probe-default_asyncintra_c4-20_0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-seqrank+bszrank_default &> wslout/output_wikitext-2v1_opt-6.7b_clm_5_128_0.6_wandasp_probe-default_asyncintra_c4-20_0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-0.5+0.1-seqrank+bszrank_default_$timestamp.txt
wait
