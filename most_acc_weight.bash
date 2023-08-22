mkdir /home/aanwar/le000288/Efficient_Inference/msiout
sbatch msiout0CIFAR10_resnet18_0.1_100_iid_fedavg_5_0_1_2_channel-wise_PQ_0.01.pbs --wait
mkdir /home/aanwar/le000288/Efficient_Inference/msiout
sbatch msiout0CIFAR10_resnet18_0.1_100_iid_fedavg_5_0_1_2_filter-wise_PQ_0.01.pbs --wait
