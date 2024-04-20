#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_llama-2-7b_clm_10_128_0.7_probe_calib-ema-probe_asyncintra_c4-200_0.1whole-0.1whole-0.1whole-0.1-0.1-bszabsnml_default &> wslout/output_wikitext-2v1_llama-2-7b_clm_10_128_0.7_probe_calib-ema-probe_asyncintra_c4-200_0.1whole-0.1whole-0.1whole-0.1-0.1-bszabsnml_default_$timestamp.txt
wait
