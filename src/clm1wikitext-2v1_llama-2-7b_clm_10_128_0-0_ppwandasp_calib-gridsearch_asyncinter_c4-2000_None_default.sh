#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model_gridsearch.py --device cuda --resume_mode 0 --init_seed 1 --control_name wikitext-2v1_llama-2-7b_clm_10_128_0-0_ppwandasp_calib-gridsearch_asyncinter_c4-2000_None_default &> wslout/output_wikitext-2v1_llama-2-7b_clm_10_128_0-0_ppwandasp_calib-gridsearch_asyncinter_c4-2000_None_default_$timestamp.txt
wait
