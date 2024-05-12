#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_local_tuned_model.py --device cuda --resume_mode 0 --init_seed 2 --control_name wikitext-2v1_llama-2-7b_clm_20_1024_0.4_None_llmpruner-tune_asyncinter_None_None_None
wait
