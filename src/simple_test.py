


# import torch


# a = torch.tensor([])

# b = [a]

# print(len(b))


# mask = torch.ones(10, dtype=torch.bool)
# print('pre_mask', mask)
# # Mark the indices to be pruned as False
# mask[None] = False
# print('mask', mask)

import torch
import torch.nn as nn

# # Example instances
# embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=3)
# linear_layer = nn.Linear(in_features=10, out_features=5)

# # Check if each instance is an instance of Embedding or Linear
# print(isinstance(embedding_layer, nn.Embedding))  # True
# print(isinstance(embedding_layer, nn.Linear))     # False

# print(isinstance(linear_layer, nn.Embedding))     # False
# print(isinstance(linear_layer, nn.Linear))        # True


# a = 'lora'

# b = a.split('-')

# print(b)
# sub_vanilla_info = ['', 0, 3.74174427986145, 124647170, 0,'zzzz']
# NUM_PARAMETER_UNIT = (1000000, 'Million')
# FLOPS_UNIT = (1000000, 'Million')
# # already in seconds unit
# TIME_UNIT = (1, 's')
# print(FLOPS_UNIT[0])
# print(f"VANILLA: {sub_vanilla_info[0]} - {sub_vanilla_info[1]/FLOPS_UNIT[0]:.2f}" , flush=True)

# print(f"VANILLA: {sub_vanilla_info[0]} - {sub_vanilla_info[1]/{FLOPS_UNIT[0]}:.2f} {FLOPS_UNIT[1]}Flops - {sub_vanilla_info[2]/TIME_UNIT[0]:.2f} {TIME_UNIT[1]} \
#               - {sub_vanilla_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - {sub_vanilla_info[4]}", flush=True)

a = torch.tensor([1,2,3,4])

print(a.numel())

b = a[0:0]
print(b)
print(b.numel())


c = torch.empty(0)
d = torch.empty(3, 0, 2)

print(c, c.numel())
print(d, d.numel())


import torch

# Define dimensions
C_out = 3  # Number of output channels
C_in = 4   # Number of input channels
N = 2      # Number of samples
L = 1      # Additional factor (for simplicity, we keep it 1)

# Define desired sparsity
s = 0.5  # 50% sparsity

# Create a random weight matrix W and input matrix X
W = torch.randn(C_out, C_in)
X = torch.randn(N * L, C_in)

# Define the pruning function with the correction
def prune(W, X, s):
    temp = X.norm(p=2, dim=0)
    print('temp', temp)
    metric = W.abs() * X.norm(p=2, dim=0)
    print('metric', metric)
    _, sorted_idx = torch.sort(metric, dim=1)
    print('sorted_idx', sorted_idx)
    pruned_idx = sorted_idx[:, :int(C_in * s)]
    print('pruned_idx', pruned_idx)
    # Create a tensor of zeros with the same shape as the pruned indices
    zeros = torch.zeros_like(W[:, :int(C_in * s)])
    W.scatter_(dim=1, index=pruned_idx, src=zeros)
    return W

# Apply the pruning function
W_pruned = prune(W, X, s)