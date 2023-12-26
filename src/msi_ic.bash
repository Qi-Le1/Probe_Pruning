mkdir /home/aanwar/le000288/Efficient_representation_inference/src/msiout
sbatch ic0CIFAR10_resnet18_ic_1_pqstructlocal:h:2:9999:1:max_inter_somemethods-3_default.pbs --wait
mkdir /home/aanwar/le000288/Efficient_representation_inference/src/msiout
sbatch ic0CIFAR100_resnet18_ic_1_pqstructlocal:h:2:9999:1:max_inter_somemethods-3_default.pbs --wait
mkdir /home/aanwar/le000288/Efficient_representation_inference/src/msiout
sbatch ic0CIFAR10_resnet18_ic_1_w*pqstructlocal:h:2:9999:1:max_inter_somemethods-3_default.pbs --wait
mkdir /home/aanwar/le000288/Efficient_representation_inference/src/msiout
sbatch ic0CIFAR100_resnet18_ic_1_w*pqstructlocal:h:2:9999:1:max_inter_somemethods-3_default.pbs --wait
