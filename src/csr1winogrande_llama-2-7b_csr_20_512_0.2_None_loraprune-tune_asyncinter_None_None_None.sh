#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_local_tuned_model.py --device cuda --resume_mode 0 --init_seed 1 --control_name winogrande_llama-2-7b_csr_20_512_0.2_None_loraprune-tune_asyncinter_None_None_None &> wslout/output_winogrande_llama-2-7b_csr_20_512_0.2_None_loraprune-tune_asyncinter_None_None_None_$timestamp.txt
wait
