#!/bin/bash
python3 make_umn_msi_commands.py --pbs_prefix csr --file test_dense_model --code_folder Probe_Pruning --res_folder msiout --run test --device cuda --num_gpus 1 --round 1 --world_size 1 --num_experiments 3 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --data csr
python3 make_umn_msi_commands.py --pbs_prefix clm --file test_dense_model --code_folder Probe_Pruning --res_folder msiout --run test --device cuda --num_gpus 1 --round 1 --world_size 1 --num_experiments 3 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --data clm
