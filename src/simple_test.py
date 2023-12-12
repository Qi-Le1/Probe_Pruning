


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

# a = torch.tensor([1,2,3,4])

# print(a.numel())

# b = a[0:0]
# print(b)
# print(b.numel())


# c = torch.empty(0)
# d = torch.empty(3, 0, 2)

# print(c, c.numel())
# print(d, d.numel())


# import torch

# # Define dimensions
# C_out = 3  # Number of output channels
# C_in = 4   # Number of input channels
# N = 2      # Number of samples
# L = 1      # Additional factor (for simplicity, we keep it 1)

# # Define desired sparsity
# s = 0.5  # 50% sparsity

# # Create a random weight matrix W and input matrix X
# W = torch.randn(C_out, C_in)
# X = torch.randn(N * L, C_in)

# # Define the pruning function with the correction
# def prune(W, X, s):
#     temp = X.norm(p=2, dim=0)
#     print('temp', temp)
#     metric = W.abs() * X.norm(p=2, dim=0)
#     print('metric', metric)
#     _, sorted_idx = torch.sort(metric, dim=1)
#     print('sorted_idx', sorted_idx)
#     pruned_idx = sorted_idx[:, :int(C_in * s)]
#     print('pruned_idx', pruned_idx)
#     # Create a tensor of zeros with the same shape as the pruned indices
#     zeros = torch.zeros_like(W[:, :int(C_in * s)])
#     W.scatter_(dim=1, index=pruned_idx, src=zeros)
#     return W

# # Apply the pruning function
# W_pruned = prune(W, X, s)

# import numpy as np

# # Define custom bin edges
# bin_edges = [
#     -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, # -1000 to -100
#     -50, 0, 50, 100,  # -100 to 100
#     200, 300, 400, 500, 600, 700, 800, 900, 1000  # 100 to 1000
# ]

# # Fine bins around -10 to 10
# fine_bins = np.arange(-10, 10, 0.1).tolist()
# bin_edges = bin_edges + fine_bins + [1e-3]

# # Sort the bin edges
# bin_edges = sorted(set(bin_edges))
# print(bin_edges)
# # Example data from a batch
# batch_data = np.random.uniform(-1000, 1000, 1000)  # Replace with your actual batch data

# # Bin the data
# hist, _ = np.histogram(batch_data, bins=bin_edges)

# # hist now contains the count of data points in each bin
# print(hist)


# import matplotlib.pyplot as plt

# # Example histogram data
# hist_data = [10, 15, 7, 12, 5]  # Frequency counts for each bin

# # Corresponding bin edges
# bin_edges = [0, 1, 2, 3, 4, 5]  # Define the range of each bin

# # Calculate the width for each bin
# bin_widths = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges)-1)]

# print(bin_edges[:-1])
# # Create the bar plot
# plt.bar(bin_edges[:-1], hist_data, width=bin_widths, align='edge')

# # Labeling
# plt.xlabel('Value Range')
# plt.ylabel('Frequency')
# plt.title('Histogram')

# # Show the plot
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# # Generate sample data
# data = np.random.normal(0, 1, 1000)

# # Compute histogram
# counts, bins = np.histogram(data, bins=30)
# counts = [100, 100, 100]
# bins = [-0.1, 0, 0.1, 0.2]
# a = (sum(counts) * np.diff(bins))
# print('a', a)
# print('counts', counts)
# print('sum(counts)', sum(counts))
# print('bins', bins),
# print('np.diff(bins)', np.diff(bins)),

# # Calculate density
# density = counts / (sum(counts) * np.diff(bins))
# print('density', density)
# # Plotting the histogram as a density
# # plt.bar(bins[:-1], density, width=np.diff(bins), edgecolor='black')
# plt.hist(data, bins, density=True, edgecolor='black')

# plt.title('Density Histogram')
# plt.xlabel('Value')
# plt.ylabel('Density')

# plt.show()


from datasets import load_dataset
from transformers import AutoTokenizer


# def load_and_tokenize_dataset(model_checkpoint, dataset_name='wikitext', dataset_version='wikitext-2-v1', max_length=512):
#     # count = 0
#     # Load the dataset
#     dataset = load_dataset(dataset_name, dataset_version, split='test')

#     # Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#     # Tokenization function
#     def tokenize_function(examples):
#         # global count
#         print('examples', examples)
#         text = examples["text"]
#         a1 = text[0]
#         a = text[1]
#         b = text[2]
#         if not text.strip():  # Check if the string is empty
#             text = "[EMPTY]" 
#         # count += 1
#         # b = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
#         return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

#     def remove_empty_examples(example):
#         return example["text"].strip() != ""

#     testenc = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
#     filtered_dataset = dataset.filter(remove_empty_examples)

#     # Tokenize the dataset
#     tokenized_dataset = filtered_dataset.map(tokenize_function, batched=True)
#     # print('count', count)
#     print('tokenized_dataset', len(tokenized_dataset))
#     # Prepare the dataset for training (for a language modeling task)
#     tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
#     tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

#     return tokenized_dataset

# # Example usage
# model_checkpoint = "gpt2"  # Replace with your model of choice
# processed_dataset = load_and_tokenize_dataset(model_checkpoint)


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = SimpleNet()


import torch

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to("cpu")

# # Flatten all weights and store their original shapes
# original_shapes = []
# all_weights = []
# for name, module in model.named_modules():
# # for name, param in model.named_parameters():
#     print('name', name)
#     # if param.requires_grad:
#     if 'fc' in name:
#         if hasattr(module, 'weight') and module.weight is not None:
#             all_weights.append(module.weight.data.view(-1))
#         elif hasattr(module, 'bias') and module.bias is not None:
#             all_weights.append(module.bias.data.view(-1))

# # Concatenate all weights and convert to a single vector
# all_weights_vector = torch.cat(all_weights)

# # Rank weights by absolute value and find the threshold for pruning
# num_weights_to_prune = int(all_weights_vector.numel() * 0.1)  # 10% Pruning
# threshold = torch.kthvalue(torch.abs(all_weights_vector), num_weights_to_prune).values

# # Apply pruning directly without a separate mask
# index = 0
# for name, module in model.named_modules():
#     if 'fc' in name:
#         if hasattr(module, 'weight') and module.weight is not None:
#             numel = module.weight.data.numel()
#             # Directly modify the weights based on the threshold
#             module.weight.data.view(-1).abs_().clamp_(min=threshold)
#             index += numel
#         elif hasattr(module, 'bias') and module.bias is not None:
#             numel = module.bias.data.numel()
#             # Directly modify the weights based on the threshold
#             module.bias.data.view(-1).abs_().clamp_(min=threshold)
#             index += numel


# model.to('cpu') 



import torch

# Move the model to CPU for processing
# model.to("cpu")

# # Step 1: Calculate norms per channel for convolutional layers
# channel_norms = []
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Conv2d):
#         # Calculate L2 norm per channel
#         norms = module.weight.data.norm(dim=(1, 2, 3), p=2)
#         for i, norm in enumerate(norms):
#             channel_norms.append((norm.item(), name, i, module.weight.shape))  # Store norm, layer name, channel index, and shape

# # Step 2: Rank these norms globally and find the threshold for pruning
# channel_norms.sort(key=lambda x: x[0])
# num_channels_to_prune = int(len(channel_norms) * 0.001)  # 0.1% pruning
# pruning_threshold = channel_norms[num_channels_to_prune - 1][0] if num_channels_to_prune > 0 else float('inf')

# # Step 3: Prune channels based on the threshold
# for norm_value, layer_name, channel_index, shape in channel_norms:
#     if norm_value > pruning_threshold:
#         break  # Stop pruning as all further channels have a norm above the threshold
#     layer = dict(model.named_modules())[layer_name]
#     # Prune the specific channel
#     with torch.no_grad():
#         layer.weight.data[channel_index] = torch.zeros(shape[1:])

# # Optionally, prune the fully connected layers based on the absolute value threshold
# all_weights = []
# for name, module in model.named_modules():
#     if 'fc' in name:
#         if hasattr(module, 'weight') and module.weight is not None:
#             all_weights.append(module.weight.data.view(-1))
#         if hasattr(module, 'bias') and module.bias is not None:
#             all_weights.append(module.bias.data.view(-1))

# # Concatenate all weights into a single vector and find the threshold for pruning
# all_weights_vector = torch.cat(all_weights)
# num_weights_to_prune = int(all_weights_vector.numel() * 0.1)  # 10% pruning
# threshold = torch.kthvalue(torch.abs(all_weights_vector), num_weights_to_prune).values

# # Apply pruning directly without a separate mask
# for name, module in model.named_modules():
#     if 'fc' in name:
#         if hasattr(module, 'weight') and module.weight is not None:
#             module.weight.data.abs_().clamp_(min=threshold)
#         if hasattr(module, 'bias') and module.bias is not None:
#             module.bias.data.abs_().clamp_(min=threshold)

# # Move the model back to CPU
# model.to('cpu')


def compare_norms_multi_dim(tensor, dims):
    # Calculate the norm using .norm()
    norm_result = tensor.norm(dim=dims)

    # Calculate the norm using torch.linalg.vector_norm()
    vector_norm_result = torch.linalg.vector_norm(tensor, dim=dims)

    # Output comparison
    print(f"Tensor shape: {tensor.shape}, Dimensions: {dims}")
    print("Result using .norm():", norm_result)
    print("Result using torch.linalg.vector_norm():", vector_norm_result)
    print("Are the results the same?:", torch.allclose(norm_result, vector_norm_result))
    print("\n")

# Create a 4D tensor with random values
tensor = torch.randn(4, 4, 4, 4)

# Compare norms across dimensions (2, 3)
compare_norms_multi_dim(tensor, dims=(2, 3))
compare_norms_multi_dim(tensor, dims=(1, 3))
compare_norms_multi_dim(tensor, dims=(1,2, 3))
