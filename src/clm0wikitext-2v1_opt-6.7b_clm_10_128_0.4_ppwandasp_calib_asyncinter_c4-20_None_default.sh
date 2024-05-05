#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 0 --control_name wikitext-2v1_opt-6.7b_clm_10_128_0.4_ppwandasp_calib_asyncinter_c4-20_None_default &> wslout/output_wikitext-2v1_opt-6.7b_clm_10_128_0.4_ppwandasp_calib_asyncinter_c4-20_None_default_$timestamp.txt
wait
