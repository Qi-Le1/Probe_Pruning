#!/bin/bash
python3 make_umn_msi_commands.py --pbs_prefix clm --file test_fix_pruned_model --code_folder Efficient_representation_inference --res_folder msiout --run test --device cuda --num_gpus 1 --round 1 --world_size 1 --num_experiments 2 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --data missing


