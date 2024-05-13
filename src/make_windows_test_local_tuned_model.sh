#!/bin/bash
python3 make_windows_commands.py --pbs_prefix clm --file test_local_tuned_model --code_folder Efficient_representation_inference --res_folder msiout --run test --device cuda --num_gpus 8 --round 1 --world_size 1 --num_experiments 3 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --data clm
# python3 make_windows_commands.py --pbs_prefix csr --file test_local_tuned_model --code_folder Efficient_representation_inference --res_folder msiout --run test --device cuda --num_gpus 8 --round 1 --world_size 1 --num_experiments 3 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --data csr
