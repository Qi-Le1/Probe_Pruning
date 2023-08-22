#!/bin/bash

# run joint
# python3 large_scale_commands_creation_msi.py --pbs_prefix c --file classifier_fl --code_folder src --res_folder test_baseline --run train --device cuda --num_gpus 1 --round 3 --world_size 1 --num_experiments 3 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --data CIFAR10
# python3 large_scale_commands_creation_msi.py --pbs_prefix c --file classifier_fl --code_folder src --res_folder test_baseline --run train --device cuda --num_gpus 1 --round 3 --world_size 1 --num_experiments 2 --experiment_step 1 --init_seed 1 --resume_mode 0 --log_interval 0.25 --data CIFAR10
# python3 large_scale_commands_creation_msi.py --pbs_prefix dp --file heuristic_search --code_folder src --res_folder dp --run train --device cuda --num_gpus 1 --round 1 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --data CIFAR10
python3 large_scale_commands_creation_msi.py --pbs_prefix msiout --file classifier_fl --code_folder Efficient_Inference --res_folder msiout --run test --device cuda --num_gpus 1 --round 1 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --data CIFAR10


# python3 large_scale_commands_creation_msi.py --pbs_prefix res --file classifier_fl --code_folder src --res_folder test_resnet_baseline --run train --device cuda --num_gpus 1 --round 3 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --data CIFAR10


# python3 large_scale_commands_creation_msi.py --pbs_prefix resnet --file classifier_fl --code_folder src --res_folder test_baseline_resnet --run train --device cuda --num_gpus 1 --round 3 --world_size 1 --num_experiments 3 --experiment_step 1 --init_seed 0 --resume_mode 1 --log_interval 0.25 --data CIFAR10
# python3 large_scale_commands_creation_msi.py --pbs_prefix b --file classifier_fl --code_folder src --res_folder test_different_portion_freq --run train --device cuda --num_gpus 1 --round 3 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --data CIFAR10

# python3 large_scale_commands_creation_msi.py --file classifier_fl --code_folder src --res_folder test_different_average --run train --device cuda --num_gpus 1 --round 6 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 1 --resume_mode 0 --log_interval 0.25 --data CIFAR10
# python3 large_scale_commands_creation_msi.py --file classifier_fl --code_folder src --res_folder test_different_average --run train --device cuda --num_gpus 1 --round 6 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 2 --resume_mode 0 --log_interval 0.25 --data CIFAR10
# python3 large_scale_commands_creation_msi.py --file classifier_fl --code_folder src --res_folder test_different_average --run train --device cuda --num_gpus 1 --round 6 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 3 --resume_mode 0 --log_interval 0.25 --data CIFAR10


# python3 large_scale_commands_creation_msi.py --pbs_prefix b --file heuristic_search --code_folder src --res_folder test_combination --run train --device cuda --num_gpus 1 --round 3 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 1 --log_interval 0.25 --data CIFAR10



