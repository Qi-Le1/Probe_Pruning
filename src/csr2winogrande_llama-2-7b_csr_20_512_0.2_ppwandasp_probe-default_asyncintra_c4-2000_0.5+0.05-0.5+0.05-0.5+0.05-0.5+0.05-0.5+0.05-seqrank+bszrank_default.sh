#!/bin/bash -l
timestamp=$(date +%Y%m%d%H%M%S)
python test_model.py --device cuda --resume_mode 0 --init_seed 2 --control_name winogrande_llama-2-7b_csr_20_512_0.2_ppwandasp_probe-default_asyncintra_c4-2000_0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank_default &> wslout/output_winogrande_llama-2-7b_csr_20_512_0.2_ppwandasp_probe-default_asyncintra_c4-2000_0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank_default_$timestamp.txt
wait
