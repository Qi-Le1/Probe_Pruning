#!/bin/bash

python make.py --mode full --task_name s2s --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 12 --split_round 1 --num_gpu 4
python make.py --mode full --task_name s2s --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 12 --split_round 1 --num_gpu 4

python make.py --mode full --task_name clm --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode full --task_name clm --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

python make.py --mode full --task_name sc --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 16 --split_round 1 --num_gpu 4
python make.py --mode full --task_name sc --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 16 --split_round 1 --num_gpu 4

python make.py --mode full --task_name ic --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 16 --split_round 1 --num_gpu 4
python make.py --mode full --task_name ic --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 16 --split_round 1 --num_gpu 4

python make.py --mode peft --task_name s2s --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 12 --split_round 1 --num_gpu 4
python make.py --mode peft --task_name s2s --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 12 --split_round 1 --num_gpu 4

python make.py --mode peft --task_name clm --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 3 --num_gpu 4
python make.py --mode peft --task_name clm --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 3 --num_gpu 4

python make.py --mode peft --task_name sc --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4
python make.py --mode peft --task_name sc --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4

python make.py --mode peft --task_name ic --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4
python make.py --mode peft --task_name ic --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4

python make.py --mode cola --task_name s2s --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 10 --split_round 1 --num_gpu 4
python make.py --mode cola --task_name s2s --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 10 --split_round 1 --num_gpu 4

python make.py --mode cola --task_name clm --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode cola --task_name clm --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

python make.py --mode cola --task_name sc --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4
python make.py --mode cola --task_name sc --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4

python make.py --mode cola --task_name ic --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4
python make.py --mode cola --task_name ic --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4

python make.py --mode cola_step --task_name s2s --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 12 --split_round 1 --num_gpu 4
python make.py --mode cola_step --task_name s2s --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 12 --split_round 1 --num_gpu 4

python make.py --mode cola_step --task_name clm --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode cola_step --task_name clm --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

python make.py --mode cola_step --task_name sc --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4
python make.py --mode cola_step --task_name sc --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4

python make.py --mode cola_step --task_name ic --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4
python make.py --mode cola_step --task_name ic --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4

python make.py --mode cola_dist --task_name clm --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode cola_dist --task_name clm --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

python make.py --mode cola_merge --task_name s2s --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 10 --split_round 1 --num_gpu 4
python make.py --mode cola_merge --task_name s2s --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 10 --split_round 1 --num_gpu 4

python make.py --mode cola_merge --task_name clm --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode cola_merge --task_name clm --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

python make.py --mode cola_merge --task_name sc --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4
python make.py --mode cola_merge --task_name sc --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4

python make.py --mode cola_merge --task_name ic --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4
python make.py --mode cola_merge --task_name ic --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 24 --split_round 1 --num_gpu 4

python make.py --mode cola_dist_merge --task_name clm --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode cola_dist_merge --task_name clm --run test --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

python make.py --mode full_dreambooth --task_name t2i --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode full_dreambooth --task_name t2i --run generate --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

python make.py --mode peft_dreambooth --task_name t2i --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode peft_dreambooth --task_name t2i --run generate --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

python make.py --mode cola_dreambooth --task_name t2i --run train --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4
python make.py --mode cola_dreambooth --task_name t2i --run generate --num_experiment 1 --resume_mode 1 --init_seed 0 --round 8 --split_round 1 --num_gpu 4

python make.py --mode computation --task_name s2s --run test --num_experiment 1 --resume_mode 0 --init_seed 0 --round 1 --split_round 65535 --num_gpu 4

python make.py --mode computation --task_name clm --run test --num_experiment 1 --resume_mode 0 --init_seed 0 --round 1 --split_round 65535 --num_gpu 4

python make.py --mode computation --task_name sc --run test --num_experiment 1 --resume_mode 0 --init_seed 0 --round 1 --split_round 65535 --num_gpu 4

python make.py --mode computation --task_name ic --run test --num_experiment 1 --resume_mode 0 --init_seed 0 --round 1 --split_round 65535 --num_gpu 4

python make.py --mode computation --task_name t2i --run test --num_experiment 1 --resume_mode 0 --init_seed 0 --round 1 --split_round 65535 --num_gpu 4

python make.py --mode computation_cola --task_name s2s --run test --num_experiment 1 --resume_mode 0 --init_seed 0 --round 1 --split_round 65535 --num_gpu 4

python make.py --mode computation_cola --task_name clm --run test --num_experiment 1 --resume_mode 0 --init_seed 0 --round 1 --split_round 65535 --num_gpu 4

python make.py --mode computation_cola --task_name sc --run test --num_experiment 1 --resume_mode 0 --init_seed 0 --round 1 --split_round 65535 --num_gpu 4

python make.py --mode computation_cola --task_name ic --run test --num_experiment 1 --resume_mode 0 --init_seed 0 --round 1 --split_round 65535 --num_gpu 4

python make.py --mode computation_cola --task_name t2i --run test --num_experiment 1 --resume_mode 0 --init_seed 0 --round 1 --split_round 65535 --num_gpu 4