


# import torch
import numpy as np

# a = torch.tensor([])

# b = [a]
import collections
# print(len(b))


# mask = torch.ones(10, dtype=torch.bool)
# print('pre_mask', mask)
# # Mark the indices to be pruned as False
# mask[None] = False
# print('mask', mask)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch

# 假设 a 和 b 是两个三维张量
# a = torch.rand(2, 3)
# b = torch.rand(3, 4)

# c = F.linear(a, b.T)
# d = torch.tensordot(a, b, dims=[[1]])
# dd = c == d
# ddd = d.shape
# # 在 a 的最后一个维度和 b 的第一个维度上进行点积
# e = 5
import torch
import torch.nn as nn

# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         # self.fc1 = nn.Linear(100, 200)
#         # self.relu = nn.ReLU()
#         # self.fc2 = nn.Linear(200, 10)

#     def forward(self, x):
#         return torch.matmul(x[0], x[1])
#         # return x[0] * x[1]

# # Instantiate the model
# model = SimpleModel()


# from deepspeed.profiling.flops_profiler import get_model_profile

# # Define a batch size and input tensor
# # batch_size = 32
# # input_tensor = torch.randn(batch_size, 100)
# x = torch.randn(2, 100)
# # y = torch.randn(100)
# # Use the profiler to get FLOPs and parameter counts
# flops, macs, params = get_model_profile(model=model, 
#                                  input_shape=(2,100))
                                

# print(f"MACs: {macs}")
# print(f"Parameters: {params}")
# c = 6
# # Example instances
# embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=3)
# linear_layer = nn.Linear(in_features=10, out_features=5)

# # Check if each instance is an instance of Embedding or Linear
# print(isinstance(embedding_layer, nn.Embedding))  # True
# print(isinstance(embedding_layer, nn.Linear))     # False

# print(isinstance(linear_layer, nn.Embedding))     # False
# print(isinstance(linear_layer, nn.Linear))        # True
# self.exclude_dim_to_aggregate = None
# self.sort_norm_dim = 0
# a = torch.tensor([5])

# # Add an extra dimension
# # To add it at the 0th dimension (making it 1x1)
# a = a.unsqueeze(0)

# # Now a is a 2D tensor with shape (1, 1)
# print(a)  # Outputs tensor([[5]])
# print(a.shape)  # Outputs torch.Size([1, 1])
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
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# input_data = np.array([0.1 for i in range(100053)])

# # input_data_length = len(input_data)
# compress_ratio = 1000
# pace = int(len(input_data) // compress_ratio)
# simplified_input_data = np.array([input_data[i] for i in range(0, len(input_data), pace)])
# x = np.array(list(range(len(simplified_input_data)+1)))
# y = np.array(list(range(len(simplified_input_data)+1)))
# x, y = np.meshgrid(x, y)
# eta = np.full(x.shape, np.nan)
# # eta = np.full(x.shape, 6, dtype=float)
# # mask = y < x
# print('eta', eta.shape)
# mask = y < x

# # Applying the mask
# x = np.where(mask, x, np.nan)  # Replace values not in the upper triangle with NaN
# y = np.where(mask, y, np.nan)

# pq_p = 1
# pq_q = 2

import torchvision.models as models
import torch


def new_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x


# define a resnet instance
resnet = models.resnet18()

# add new_forward function to the resnet instance as a class method
bound_method = new_forward.__get__(resnet, resnet.__class__)
setattr(resnet, 'forward', bound_method)
aa = resnet.forward
a = 5
# # print(len(x), len(x[0]))
# for d in range(1, len(x)):
#     # m at most equals to d-1
#     cur_dimension = d * pace
#     pq_index = simplified_input_data[d-1]
#     for m in range(1, d):
#     # for m in range(1, len(x[0])):
#         cur_rest_dimension = m * pace

#         sub_eta = ((cur_rest_dimension / (((1 - pq_index) ** (pq_q * pq_p / (pq_q - pq_p))) * cur_dimension)) ** (-(pq_q - pq_p) / pq_q)) - 1
#         # print('sub_eta', sub_eta)
#         # print(d, m, sub_eta)
#         if sub_eta < -1:
#             sub_eta = -1
#         elif sub_eta > 2:
#             sub_eta = 2
#         # print(type(sub_eta))

#         # print('d', d, 'm', m, 'sub_eta', sub_eta, type(d), type(m))
#         eta[m][d] = sub_eta

# # for i in range(1, len(x[0])):
# #     # fix = i
# #     # for j in range(1, fix):
# #     for j in range(1, len(x[0])):
# #         eta[i][j] = 3

# z = np.sin(np.sqrt(x**2 + y**2))
# print('z', z.shape)
# # Create a figure and a 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# print('eta', eta)
# # Plot a 3D surface
# surf = ax.plot_surface(x, y, eta, cmap='viridis')
# # surf = ax.plot_surface(x, y, z, cmap='viridis')
# # Add a color bar which maps values to colors
# fig.colorbar(surf, shrink=0.5, aspect=5)

# ax.set_title('3D Heatmap')
# ax.set_xlabel('d dimension')
# ax.set_ylabel('m dimension')
# ax.set_zlabel('eta')
# plt.show()

a = 5
# a = [[0.40625],
#  [0.4375 ]]

# b = np.std(a, axis=1)
# c = np.std(a, axis=0)
# d = 5
# c = torch.empty(0)
# d = torch.empty(3, 0, 2)

# print(c, c.numel())
# print(d, d.numel())

# a = torch.tensor([[1,2,5,9,10], [1,3, 8, 15, 20]], dtype=torch.float32)
# standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)
# b = a ** 2
# c = standarlization(a)
# d = standarlization(b)
# e = 5
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import torch
import psutil
import os

def memory_usage_in_MB():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Memory in MB

# Memory usage before the operation


# Your PyTorch code
a = torch.randn(40, 1100)
b = torch.randn(1100, 40)

temp_a = copy.deepcopy(a)
temp_b = copy.deepcopy(b)
memory_before = memory_usage_in_MB()
c = torch.matmul(a, b)

# Memory usage after the operation
memory_after = memory_usage_in_MB()

# Calculate the difference
memory_consumed = memory_after - memory_before
print(f"Memory consumed: {memory_consumed:.2f} MB")


memory_before = memory_usage_in_MB()
a = a.unsqueeze(-1)
b = b.unsqueeze(0)
result = a * b
print('result', result.shape)
memory_after = memory_usage_in_MB()

# Calculate the difference
memory_consumed = memory_after - memory_before
print(f"Memory consumed broadcast: {memory_consumed:.2f} MB")


memory_before = memory_usage_in_MB()
result = (a * b).sum(dim=(0,2))
print('result', result.shape)

# result2 = a.sum(dim=0) * b.sum(dim=1)
# print('result2', result2.shape)

result3 = temp_a.sum(0) * temp_b.sum(1)
print('result3', result3.shape)
dd = result == result3

result4 = (temp_a * temp_b.sum(1)).sum(0)
ddd = result == result4
memory_after = memory_usage_in_MB()

# Calculate the difference
memory_consumed = memory_after - memory_before
print(f"Memory consumed 2: {memory_consumed:.2f} MB")

a = 6
# Settings for the network
# layer_sizes = [5, 8, 8, 5]  # Example layer sizes
# n_layers = len(layer_sizes)
# layer_positions = range(n_layers)
# node_radius = 0.05

# # Create the plot
# fig, ax = plt.subplots()

# # Function to draw nodes
# def draw_layer(y, size, label):
#     x_values = [x * 0.2 for x in range(size)]
#     for x in x_values:
#         circle = plt.Circle((x, y), node_radius, color='blue', fill=True)
#         ax.add_artist(circle)
#     # Optionally add a label for the layer
#     ax.text(x_values[-1] + 0.15, y, label, fontsize=12)

# # Draw the layers
# for i, size in enumerate(layer_sizes):
#     draw_layer(layer_positions[i], size, f'Layer {i+1}')

# # Highlight the middle layer
# middle_layer_index = n_layers // 2 - 1
# highlight_rect = patches.Rectangle((-0.1, middle_layer_index - 0.1), 
#                                    layer_sizes[middle_layer_index] * 0.2, 
#                                    0.3, linewidth=2, edgecolor='r', facecolor='none')
# ax.add_patch(highlight_rect)

# # Annotate
# ax.annotate('Our theory-guided adaptive pruning', 
#             xy=(layer_sizes[middle_layer_index] * 0.1, middle_layer_index), 
#             xytext=(layer_sizes[middle_layer_index] * 0.5, middle_layer_index + 1),
#             arrowprops=dict(facecolor='black', shrink=0.05))

# # Set the limits and labels
# ax.set_xlim(-0.2, max(layer_sizes) * 0.2)
# ax.set_ylim(-1, n_layers)
# ax.set_aspect('equal', adjustable='datalim')
# ax.axis('off')
a = torch.tensor([[0.6815, 0.7796, 0.9360, 0.5866, 1.8860],
 [0.1141, 0.1273, 0.4898, 1.0005, 0.2570],
 [0.2012, 0.2757, 0.2001, 1.2834, 0.4445]])
b = a / torch.sum(a, axis=-1, keepdim=True)
print('a', a, 'b', b)
# class Fulei:
#     def __init__(self):
#         pass

#     def fuleicall(self):
#         print(self.weight)

# class zilei(Fulei):
#     def __init__(self):
#         super().__init__()
#         self.weight = 16

#     def zileicall(self):
#         self.fuleicall()


# a = zilei()
# a.zileicall()
# b = 5

# a = torch.tensor([[1, 2, 3, 4, 5], [6,7,8,9,10]])

# b = torch.tensor([[11, 12, 13], [16,17,18]])

# a[..., [1,2,3]] = b

# print(a[0], a[1])
# c = 5
# plt.show()

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

import copy
from datasets import load_dataset
from transformers import AutoTokenizer

# for x in range(2, 30, 10):
#     print(x)
# eta = 0
# pq_p = 1
# pq_q = 2
# prune_norm = 2
# beta = 0.9
# gamma = 1
import time
# class YourClass:
#     # Other methods...
#     def __init__(self):
#         self.logger_info_time_used = 0

#     def monitor_time(func):
#         def wrapper(*args, **kwargs):
#             print('wrapper', args, kwargs)
#             start_time = time.time()
#             result = func(*args, **kwargs)
#             args[0].logger_info_time_used += time.time() - start_time
#             return result
#         return wrapper
    
#     @monitor_time
#     def update_pruning_info(self, info):
#         a = 5

# a = YourClass()
# a.update_pruning_info(1)
# def calculate_entropy(probabilities):
#     """
#     Calculate the entropy of a probability distribution.
#     :param probabilities: list of probabilities for each event
#     :return: entropy of the distribution
#     """
#     entropy = 0
#     for p in probabilities:
#         if p > 0:  # To avoid math domain error for log(0)
#             entropy -= p * math.log(p, 2)  # Log base 2 for entropy in bits
#     return entropy

# def pq_struct(w, key, prune_dim):

#     calc_dim = 1
#     # i != prune_dim and 
#     # dims_to_aggregate = tuple(i for i in range(w.dim()) if i != 0)
#     # norm_across_other_dims = torch.linalg.vector_norm(w, ord=prune_norm, dim=dims_to_aggregate)     
#     print(w)
#     norm_p = torch.linalg.vector_norm(w, ord=pq_p, dim=calc_dim)
#     norm_q = torch.linalg.vector_norm(w, ord=pq_p, dim=calc_dim) + 1e-10
    
#     dimension = w.shape[prune_dim]
#     pq_indices = (1 - dimension ** (1/pq_q - 1/pq_p) * norm_p / norm_q)

#     # add additional dimension if dimension is 0
#     if pq_indices.dim() == 0:
#         pq_indices = pq_indices.unsqueeze(0)

#     if torch.isnan(pq_indices).any():
#         raise ValueError('pq_indices contains nan values')

#     lower_bound = dimension * (1 + eta) ** (-pq_q / (pq_q - pq_p)) * (1 - pq_indices) ** (pq_q * pq_p / (pq_q - pq_p))
#     beta_tensor = torch.full_like(lower_bound, beta)
#     prune_channels_count = torch.floor(dimension * torch.min(gamma * (1 - lower_bound / dimension), beta_tensor))

#     _, sorted_channels = torch.sort(norm_across_other_dims, dim=calc_dim)
#     prune_channels = sorted_channels[:int(prune_channels_count.item())]
#     # info = {
#     #     f"{key}_norm_across_other_dims": norm_across_other_dims.mean(dim=0).squeeze(0).tolist(),
#     #     f"{key}_pq_indices": pq_indices.mean(dim=0).squeeze(0).tolist(),
#     # }
#     # self.update_pruning_info(info)
#     return prune_channels

# tensor1 = torch.tensor([1, 1, 1, 1, 1, 10, 10, 10, 10, 10])
# tensor2 = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 100])
# # sum_tensor2 = tensor1.sum()
# import math

# # Now you can use log from the math module
# print(-math.log(0.1))

# Normalize tensor2 by dividing each element by the sum
# normalized_tensor2 = tensor1 / sum_tensor2
# Stack the tensors to create a batched tensor
# The resulting tensor will have shape [2, 10]
# batched_tensor = torch.stack([tensor1, tensor2])
# batched_tensor = torch.stack([tensor1, normalized_tensor2])
# pq_struct(batched_tensor, 'w', 1)

import numpy as np

save_format = 'png'
fig_name = 'zz'
# Define the base directory for visualization
vis_path = './output/vis/{}'.format(save_format)

# Construct the full path for the figure
fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
print(vis_path, fig_path)
# Create a sample matrix X (e.g., 4 samples, 3 features)
# X = np.array([[1, 2, 3],
#               [4, 5, 7],
#               [7, 9, 9],
#               [10, 16, 12]])

# # Compute X^T X
# XTX = np.dot(X.T, X)

# # Approach 1: Full matrix inversion of X^T X, then extract diagonal
# inv_XTX = np.linalg.inv(XTX)
# diag_inv_XTX = np.diag(inv_XTX)

# # Approach 2: Extract the diagonal of X^T X, then invert each element
# diag_XTX = np.diag(XTX)
# inv_diag_XTX = 1 / diag_XTX

# # Display the results
# print("Diagonal of (X^T X)^-1:\n", diag_inv_XTX)
# print("Inverse of the diagonal elements of X^T X:\n", inv_diag_XTX)

a = 5
# a = 5
# def preprocess_function_test(dataset):
#     all_text = "\n\n".join(dataset['text'])
#     model_inputs = tokenizer(all_text, return_tensors='pt', truncation=False, padding=False)

#     max_length = 512  # Set your desired max length
#     input_ids = model_inputs['input_ids'][0]  # Assuming a single concatenated string
#     attention_mask = model_inputs['attention_mask'][0]

#     input_chunks = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length)]
#     mask_chunks = [attention_mask[i:i + max_length] for i in range(0, len(attention_mask), max_length)]

#     final_inputs = []
#     for chunk in input_chunks:
#         final_inputs.append({
#             'input_ids': torch.tensor(chunk, dtype=torch.long),
#             'attention_mask': torch.tensor(mask_chunks[final_inputs.index(chunk)], dtype=torch.long)
#         })

#     # Add labels if required
#     for item in final_inputs:
#         item['labels'] = item['input_ids'].clone()

#     return final_inputs

# def load_and_tokenize_dataset(model_checkpoint, dataset_name='wikitext', dataset_version='wikitext-2-v1', max_length=512):
#     # count = 0
#     # Load the dataset
#     dataset = load_dataset(dataset_name, dataset_version, split='test')

#     a = dataset['text']
#     # Load the tokenizer    
#     tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
    # Tokenization function
    # def preprocess_function_test(examples):
    #     max_length = 120
    #     text_column = ['text']
    #     batch_size = len(examples[text_column[0]])
        
    #     model_inputs = tokenizer("\n\n".join(examples['text']), return_tensors='pt')
    #     labels = tokenizer("\n\n".join(examples['text']), return_tensors='pt')
    #     nsamples = model_inputs["input_ids"].numel() // max_length

    #     for i in range(nsamples):
    #         start = i * max_length
    #         end = start + max_length
    #         sample_input_ids = model_inputs["input_ids"][:, start:end]
    #         sample_attention_mask = model_inputs["attention_mask"][:, start:end]
    #         # label_input_ids = labels[i]
    #         sample_input_ids.reshape(1, max_length)
    #         sample_attention_mask.reshape(1, max_length)
    #         model_inputs["input_ids"][i] = sample_input_ids
    #         model_inputs["attention_mask"][i] = sample_attention_mask
    #         # labels["input_ids"][i] = label_input_ids
    #         # model_inputs["split"].append(cfg['task_label'][examples['category'][i]])
    #         # model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][-max_length:])
    #         # model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][-max_length:])
    #         labels["input_ids"][i] = sample_input_ids
    #         labels["attention_mask"][i] = sample_attention_mask
    #     model_inputs["labels"] = labels["input_ids"]


    #         # Tokenize all examples
    #     model_inputs = tokenizer("\n\n".join(examples['text']), max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    #     a = model_inputs["input_ids"].numel()
    #     # In this example, labels are the same as the input. Modify as needed.
    #     labels = tokenizer("\n\n".join(examples['text']), max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

    #     # The input_ids and attention_mask are already in the desired format
    #     model_inputs["labels"] = labels["input_ids"]

    #     return model_inputs

    # def remove_empty_examples(example):
    #     return example["text"].strip() != ""

    # model_inputs = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
    # labels = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
    # nsamples = model_inputs["input_ids"].numel() // max_length
    # for i in range(nsamples):
    #     start = i * max_length
    #     end = start + max_length
    #     sample_input_ids = model_inputs["input_ids"][:, start:end]
    #     sample_attention_mask = model_inputs["attention_mask"][:, start:end]
    #     # label_input_ids = labels[i]
    #     sample_input_ids.reshape(1, max_length)
    #     sample_attention_mask.reshape(1, max_length)
    #     model_inputs["input_ids"][i] = sample_input_ids
    #     model_inputs["attention_mask"][i] = sample_attention_mask
    #     # labels["input_ids"][i] = label_input_ids
    #     # model_inputs["split"].append(cfg['task_label'][examples['category'][i]])
    #     # model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][-max_length:])
    #     # model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][-max_length:])
    #     labels["input_ids"][i] = sample_input_ids
    #     labels["attention_mask"][i] = sample_attention_mask
    # model_inputs["labels"] = labels["input_ids"]
    # # testenc = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
    # # filtered_dataset = dataset.filter(remove_empty_examples)
    # # testenc = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
    # # Tokenize the dataset

#     def preprocess_function_test(examples):
#         pass
#         return
    
#     # tokenized_dataset = dataset.map(preprocess_function_test, remove_columns=dataset.column_names, batched=False)
#     a= dataset['text']
    
#     all_text = "\n\n".join(dataset['text'])
#     model_inputs = tokenizer(all_text, return_tensors='pt', truncation=False, padding=False)

#     max_length = 1024  # Set your desired max length
#     input_ids = model_inputs['input_ids'][0]  # Assuming a single concatenated string
#     attention_mask = model_inputs['attention_mask'][0]

#     input_chunks = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length)]
#     mask_chunks = [attention_mask[i:i + max_length] for i in range(0, len(attention_mask), max_length)]

#     final_inputs = collections.defaultdict(list)
#     for i in range(len(input_chunks)):
#         if len(input_chunks[i]) == max_length:
#             final_inputs['input_ids'].append(input_chunks[i])
#             final_inputs['attention_mask'].append(mask_chunks[i])
#             final_inputs['labels'].append(input_chunks[i])
    
#     dataset.remove_columns(dataset.column_names)
#     dataset['input_ids'] = final_inputs['input_ids']
#     dataset['attention_mask'] = final_inputs['attention_mask']
#     dataset['labels'] = final_inputs['labels']

#     processed_dataset = {}
#     processed_dataset['test'] = dataset.map(
#         preprocess_function_test,
#         batched=False,
#         batch_size=60,
#         num_proc=1,
#         remove_columns=dataset.column_names,
#         load_from_cache_file=False,
#         desc="Running tokenizer on dataset",
#     )
# # print('count', count)
#     # print('tokenized_dataset', len(tokenized_dataset))
#     # Prepare the dataset for training (for a language modeling task)
#     # tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
#     # tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

#     return processed_dataset['test']

# # # Example usage
# model_checkpoint = "gpt2"  # Replace with your model of choice
# processed_dataset = load_and_tokenize_dataset(model_checkpoint)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# # Define a simple neural network
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.fc1 = nn.Linear(784, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Instantiate the model
# model = SimpleNet()


import torch

# norm_across_other_dims = torch.tensor([[1, 2, 3], []])
# print(norm_across_other_dims, norm_across_other_dims.shape)
# norm_across_other_dims = norm_across_other_dims.float()
# # Calculate the mean across the zeroth dimension
# mean_tensor = norm_across_other_dims.mean(dim=0)

# # Squeeze the zeroth dimension if it's of size 1
# squeezed_tensor = mean_tensor.squeeze(0)

# # Convert the tensor to a list
# result_list = squeezed_tensor.tolist()

# print('11', result_list)



# import numpy as np

# def compare_1d_vector_norms(v, p, q, gamma, beta):
#     pq_p = p
#     pq_q = q
#     p_norm = np.linalg.norm(v, p)
#     q_norm = np.linalg.norm(v, q)

#     print(f"  {p} Norm: {p_norm}")
#     print(f"  {q} Norm: {q_norm}")

#     # Calculate and compare ratios of norms
#     dimension = len(v)
#     ratio = p_norm / q_norm

#     print(f"  {p}/{q} Norm Ratio: {ratio}", len(v) ** (1/q - 1/p))
    
#     pq_indices = (1 - dimension ** (1/q - 1/p) * p_norm / q_norm)
#     # pq_indices = 0.08
#     # dimension = 8000
#     print('scaling', dimension ** (1/q - 1/p))
#     print(f"  pq_indices_{p}_{q}: {pq_indices}")
#     eta = 0
#     lower_bound = dimension * (1 + eta) ** (-pq_q / (pq_q - pq_p)) * (1 - pq_indices) ** (pq_q * pq_p / (pq_q - pq_p))
#     beta_array = np.full_like(lower_bound, beta)
#     prune_channels_count = np.floor(dimension * np.minimum(gamma * (1 - lower_bound / dimension), beta_array))
#     print(f"ratio", {gamma * (1 - lower_bound / dimension)})
#     print(f"  Lower Bound: {lower_bound}")
#     print(f"  Prune Channels Count: {prune_channels_count}\n")
#     return pq_indices, lower_bound


# def compare_1d_vector_norms_si(v, p, q, gamma, beta):
#     pq_p = p
#     pq_q = q
#     p_norm = np.linalg.norm(v, p)
#     q_norm = np.linalg.norm(v, q)

#     print(f"  {p} Norm: {p_norm}")
#     print(f"  {q} Norm: {q_norm}")

#     # Calculate and compare ratios of norms
#     dimension = len(v)
#     ratio = p_norm / q_norm

#     print(f"  {p}/{q} Norm Ratio: {ratio}", len(v) ** (1/q - 1/p))
    
#     si_indices = p_norm / q_norm
#     # pq_indices = 0.08
#     # dimension = 8000
#     print('scaling', dimension ** (1/q - 1/p))
#     print(f"  pq_indices_{p}_{q}: {si_indices}")
#     eta = 0
#     lower_bound = si_indices ** (-pq_q / (1 - pq_q)) * (1 + eta) ** (-1 / (1 - pq_q))
#     beta_array = np.full_like(lower_bound, beta)
#     prune_channels_count = np.floor(dimension * np.minimum(gamma * (1 - lower_bound / dimension), beta_array))
#     print(f"ratio", {gamma * (1 - lower_bound / dimension)})
#     print(f"  Lower Bound: {lower_bound}")
#     print(f"  Prune Channels Count: {prune_channels_count}\n")
#     return si_indices, lower_bound
# # # pq_p = 1
# # # pq_q = 2
# gamma = 1
# beta = 0.9
# # # Example usage
# vectors = [
#     # np.array([1, 1, 1,1, 0]),
#     # np.array([1, 1, 1,1, 0,1, 1, 1,1, 0]),
#     # np.array([0.1,0.1,0.1,0.1,0.1,10]),
#     np.array([0.1 for i in range(1000)] + [100]),
#     # np.array([100,100,100,100,100,]),
#     # np.array([100,100,100,100,0]),
#     # np.array([-7, 8, -9])
#     np.random.normal(loc=0, scale=1, size=1000),
#     np.array(list(np.random.normal(loc=0, scale=1, size=1000)) + [20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 1000])
# ]
# #  (1,3), (2,3)
# pq_indices_list = []
# lower_bound_list = []
# p_values = []
# color = ['blue', 'green', 'red']
# # norm_comb = []
# norm_comb = [(p, 2) for p in np.arange(0.02, 1.01, 0.02)]
# norm_comb = [(p, 1) for p in np.arange(0.02, 1.01, 0.02)]
# # for i in range(2, len(vectors)):
# i = 2
# vector = vectors[i]
# for comb in norm_comb:
#     p = comb[0]
#     q = comb[1]
#     if p < q:
#         pq_indices, lower_bound = compare_1d_vector_norms(vector, p, q, gamma, beta)
#     else:
#         # p, q = q, p
#         pq_indices, lower_bound = compare_1d_vector_norms_si(vector, q, p, gamma, beta)
#     pq_indices_list.append(pq_indices)
#     lower_bound_list.append(lower_bound)
#     p_values.append(p)

# # Plotting
#     plt.figure(num=f'PQ Indices - Vector {i}', figsize=(10, 5))
#     plt.plot(p_values, pq_indices_list, label=f'PQ Indices - Vector {i}', color=color[i])
#     plt.xlabel('p value')
#     plt.ylabel('PQ Index Value')
#     plt.title('PQ Indices for Different p Values')
#     plt.legend()
#     plt.savefig(f'./simple_test_folder/PQ_Indices_Graph_{i}.png')  # Saving the graph

#     # Graph for Lower Bound
#     plt.figure(num=f'Lower Bound - Vector {i}', figsize=(10, 5))
#     plt.plot(p_values, lower_bound_list, label=f'Lower Bound - Vector {i}', color=color[i])
#     plt.xlabel('p value')
#     plt.ylabel('Lower Bound Value')
#     plt.title('Lower Bound for Different p Values')
#     plt.legend()
#     plt.savefig(f'./simple_test_folder/Lower_Bound_Graph_{i}.png')


# z = 6
# import numpy as np
# import matplotlib.pyplot as plt

# # Define the PQ Index calculation
# def pq_index(w, p, q):
#     norm_p = np.linalg.norm(w, p)
#     norm_q = np.linalg.norm(w, q)
#     d = len(w)
#     return 1 - d ** (1/q - 1/p) * (norm_p / norm_q)

# # Create a Gaussian distribution
# length = 1000
# data = np.random.normal(size=length)
# data = np.abs(data)
# # Sort the data
# sorted_data = np.sort(data)
# print(sorted_data)
# # Calculate PQ Index for increasing length
# p = 1  # Example value for p
# q = 2    # Example value for q
# pq_indices = [pq_index(sorted_data[:i], p, q) for i in range(1, length + 1)]
# # print(pq_indices)
# # # Plot the trend of PQ Index
# # plt.plot(pq_indices)
# # plt.xlabel('Length of the vector')
# # plt.ylabel('PQ Index')
# # plt.title('PQ Index Trend for Sorted Gaussian Distribution')
# # plt.show()




# def parallel_cal_varying_length_norm(sorted_norm, norm):
#     if norm == 1:
#         # Take the absolute value of each element
#         processed_channels = sorted_norm.abs()
#         varying_vector_norm = processed_channels.cumsum(dim=1)
#     elif norm == 2:
#         # Take the square of each element
#         processed_channels = sorted_norm.pow(2)
#         # print('processed_channels', processed_channels.shape, processed_channels[0])
#         varying_vector_norm = processed_channels.cumsum(dim=1).sqrt()
#         # print('varying_vector_norm', varying_vector_norm.shape, varying_vector_norm[0])
#     else:
#         # Handle other cases or throw an error
#         raise ValueError('Not valid norm')
#     return varying_vector_norm

# def parallel_cal_varying_length_info(sorted_norm, reversed=False):
#     if reversed:
#         sorted_norm = torch.flip(sorted_norm, [1])
#     nominator_varying_vector_norm = parallel_cal_varying_length_norm(sorted_norm, p)
#     denominator_varying_vector_norm = parallel_cal_varying_length_norm(sorted_norm, q)

#     # nominator_varying_vector_norm = nominator_varying_vector_norm.to(cfg['device'])
#     # denominator_varying_vector_norm = denominator_varying_vector_norm.to(cfg['device'])
#     # print('nominator_varying_vector_norm', nominator_varying_vector_norm.shape, nominator_varying_vector_norm[0])
#     # print('denominator_varying_vector_norm', denominator_varying_vector_norm.shape, denominator_varying_vector_norm[0])

#     num_rows, num_cols = nominator_varying_vector_norm.shape

#     # if reversed:
#     #     # Create a tensor where each row starts from 1 and decreases to the length of the row
#     #     dimension = torch.arange(num_cols, 0, -1).unsqueeze(0)
#     # else:
#         # Create a tensor where each row starts from 1 and increases to the length of the row
#     dimension = torch.arange(1, num_cols + 1).unsqueeze(0)
#     # dimension = dimension.expand(num_rows, -1).to(cfg['device'])
#     return nominator_varying_vector_norm, denominator_varying_vector_norm, dimension

# sorted_data = torch.from_numpy(sorted_data)
# sorted_data.unsqueeze_(0)

# nominator_varying_vector_norm, denominator_varying_vector_norm, dimension = parallel_cal_varying_length_info(sorted_data)
# # print('dimension', dimension.shape, dimension)
# pq_indices_varying_length = (1 - dimension ** (1/q - 1/p) * (nominator_varying_vector_norm / denominator_varying_vector_norm))

# pq_indices_varying_length = pq_indices_varying_length[0].tolist()

# plt.plot(pq_indices_varying_length)
# plt.xlabel('Length of the vector')
# plt.ylabel('PQ Index')
# plt.title('PQ Index Trend for Sorted Gaussian Distribution')
# plt.show()

# model_args = 'meta-llama/Llama-2-7b-hf'
# from lm_eval import evaluator
# results = evaluator.simple_evaluate(
#     model="hf-causal-experimental",
#     model_args=model_args,
#     tasks=task_names,
#     num_fewshot=num_fewshot,
#     batch_size=None,
#     device=None,
#     no_cache=True,
#     limit=limit,
#     description_dict={},
#     decontamination_ngrams_path=None,
#     check_integrity=False,
#     pretrained_model=model,
#     tokenizer=tokenizer, 
#     add_special_tokens=add_special_tokens
# )

# a = [[1,2,3,4], [5,6], [7]]
# b = torch.cat(a, dim=0)
# print('b', b)

# loss_fct = CrossEntropyLoss()
# a = [[0.5,0.4,0.3]]
# b = [[0.5, 0.4, 0.3]]
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
# generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
# print('generation_output', generation_output)
# a = nn.Linear(5, 10, bias=False)
# print('a.bias', a.bias)
# b = nn.Linear(5, 10, bias=True)
# print('b.bias', b.bias)
# standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

# attn_metric_list = [-0.2544, -0.4708, 1.1181, 0.9859, -0.0471]
# mlp_metric_list = [-0.5982, 0.6976, 1.2438, -0.3796, 0.2794]

# attn_metric = torch.stack(attn_metric_list)
# attn_metric = standarlization(attn_metric)
# attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)

# mlp_metric = torch.stack(mlp_metric_list)
# mlp_metric = standarlization(mlp_metric)
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, \
    AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForMultipleChoice, AutoModel
# traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
# testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

# # # Encode datasets
# tokenizer = LlamaTokenizer.from_pretrained('output/llama-2-7b',
#                                                    padding_side='left')
# trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
# temp = trainenc.input_ids.shape
# a = 5

# a = [1,2,3]
# b = torch.tensor(a)
# c = torch.mean(b)
# d = 6
# # print(pq_indices[:10])
# print(pq_indices_varying_length[0][:10])
# def compare_1d_vector_norms(v, p, q, gamma, beta, pq_indices):
#     pq_p = p
#     pq_q = q
#     # print(f"  p Norm: {p_norm}, q Norm: {q_norm}")
#     # p_norm = np.linalg.norm(v, p)
#     # q_norm = np.linalg.norm(v, q)

#     # print(f"  {p} Norm: {p_norm}")
#     # print(f"  {q} Norm: {q_norm}")

#     # # Calculate and compare ratios of norms
#     # dimension = len(v)
#     # ratio = p_norm / q_norm

#     # print(f"  {p}/{q} Norm Ratio: {ratio}", len(v) ** (1/q - 1/p))
#     dimension = 100
#     # pq_indices = (1 - dimension ** (1/q - 1/p) * p_norm / q_norm)
#     print(f"  pq_indices: {pq_indices}")
    
#     lower_bound = dimension * (1 + eta) ** (-pq_q / (pq_q - pq_p)) * (1 - pq_indices) ** (pq_q * pq_p / (pq_q - pq_p))
#     beta_array = np.full_like(lower_bound, beta)
#     prune_channels_count = np.floor(dimension * np.minimum(gamma * (1 - lower_bound / dimension), beta_array))
#     print(f"ratio", {gamma * (1 - lower_bound / dimension)})
#     print(f"  Lower Bound: {lower_bound}")
#     print(f"  Prune Channels Count: {prune_channels_count}\n")


# slopes = torch.tensor([-10, 1, 1, 1, 1, 10, 10, 10, 10, 10],  dtype=torch.float32)

#     # if 'low' in cfg['prune_name']:
#         # Avoid edge case of slope
# window_size = 4  # 10 neighbors on each side + the element itself

# # Create a window with equal weights
# window = torch.ones(window_size, dtype=torch.float32) / window_size
# # window = window.to(lower_bound.device)  # Ensure window is on the same device as lower_bound

# # Calculate the moving average using convolution
# # PyTorch's conv1d expects a 3D tensor (batch, channel, length), so we need to add extra dimensions
# slopes = slopes.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
# window = window.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

# # Use conv1d for moving average
# averages = torch.nn.functional.conv1d(slopes, window, padding=int(window_size//2))
# averages = averages.squeeze()  # 
# negative_values = averages[averages < 0]

# # Check if there are any negative values
# if len(negative_values) > 0:
#     # Find the maximum among the negative values (closest to zero)
#     closest_negative = torch.max(negative_values)
#     b = torch.where(averages == closest_negative)
#     # Get the index of this value in the original 'averages' tensor
#     first_phase_transition = torch.where(averages == closest_negative)[0][0]
# else:
#     first_phase_transition = None  # or handle the case where there are no negative values
#     raise ValueError('No negative values found in averages')

# pq_p = 1



load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], split='validation')



# # pq_q = 2
# gamma = 1
# beta = 0.9
# # Example usage
# # vectors = [
# #     np.array([1, 1, 1,1, 0]),
# #     np.array([1, 1, 1,1, 0,1, 1, 1,1, 0]),
# #     np.array([0.1,0.1,0.1,0.1,0.1,]),
# #     np.array([100,100,100,100,100,]),
# #     np.array([100,100,100,100,0]),
# #     # np.array([-7, 8, -9])
# # ]
# #  (1,3), (2,3)
# for i in np.arange(0, 1, 0.01):
#     # for j in np.arange(0.1, 1, 0.1):
#     # for comb in [(1,2)]:
#     p = 1
#     q = 2
#     vector = None
#     compare_1d_vector_norms(vector, p, q, gamma, beta, i)

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

a = torch.randn((3, 5))
# torch.sum((torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1))).reshape(-1, 1, 1) * torch.linalg.vector_norm(subset[name].weight.data, ord=1, dim=1)), dim=1)
b = torch.linalg.vector_norm(a, ord=1, dim=1)
c = 5
# def compare_norms_multi_dim(tensor, dims):
#     # Calculate the norm using .norm()
#     norm_result = tensor.norm(dim=dims)

#     # Calculate the norm using torch.linalg.vector_norm()
#     vector_norm_result = torch.linalg.vector_norm(tensor, dim=dims)

#     # Output comparison
#     print(f"Tensor shape: {tensor.shape}, Dimensions: {dims}")
#     print("Result using .norm():", norm_result)
#     print("Result using torch.linalg.vector_norm():", vector_norm_result)
#     print("Are the results the same?:", torch.allclose(norm_result, vector_norm_result))
#     print("\n")

# # Create a 4D tensor with random values
# tensor = torch.randn(4, 4, 4, 4)

# # Compare norms across dimensions (2, 3)
# compare_norms_multi_dim(tensor, dims=(2, 3))
# compare_norms_multi_dim(tensor, dims=(1, 3))
# compare_norms_multi_dim(tensor, dims=(1,2, 3))


# tensor = torch.tensor([4,5,6,7]) 
# a0 = tensor.dim()
# a = tensor.shape
# # a1 = a[0]
# b = tensor.unsqueeze(0)
# c = b.shape
# c = 5


import torch
from torch.nn import CrossEntropyLoss


a = nn.Parameter(torch.randn(3, 4))
print(a.dim(), a.shape)
a.data = torch.randn(3, 4)
print(a)
# Example logits and labels
# lm_logits = torch.randn(2, 5, 10)  # Batch size = 2, Sequence length = 5, Num classes = 10
# labels = torch.randint(0, 10, (2, 5))  # Random true labels

# # Shift logits and labels
# shift_logits = lm_logits[..., :-1, :].contiguous()  # Shape: [2, 4, 10]
# shift_labels = labels[..., 1:].contiguous()  # Shape: [2, 4]

# a = shift_logits.view(-1, shift_logits.size(-1)) # Shape: [8, 10]
# b = shift_labels.view(-1) # Shape: [8]

# # Calculate loss
# loss_fct = CrossEntropyLoss()
# loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# print("Loss:", loss.item())
# batch_size = 4
# attention_mask = torch.randn(10,4)
# if attention_mask is not None:
#     attention_mask = attention_mask.view(batch_size, -1)
#     # We create a 3D attention mask from a 2D tensor mask.
#     # Sizes are [batch_size, 1, 1, to_seq_length]
#     # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
#     # this attention mask is more simple than the triangular masking of causal attention
#     # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
#     attention_mask = attention_mask[:, None, None, :]

# c = 5

'''
attention_mask: [batch_size, num_heads, from_seq_length, to_seq_length]
kv: [batch_size, num_head, sql_len, head_features]

'''


# class GPT2Model(GPT2PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
 
#         self.wte = nn.Embedding(config.vocab_size, config.n_embd)
#         self.wpe = nn.Embedding(config.n_positions, config.n_embd)
#         self.drop = nn.Dropout(config.embd_pdrop)
#         self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
#         self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
 
#         self.init_weights()
 
#     def get_input_embeddings(self):
#         return self.wte
 
#     def set_input_embeddings(self, new_embeddings):
#         self.wte = new_embeddings
 
#     def _prune_heads(self, heads_to_prune):
#         """
#         Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
#         """
#         for layer, heads in heads_to_prune.items():
#             self.h[layer].attn.prune_heads(heads)
 
#     @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint="gpt2",
#         output_type=BaseModelOutputWithPastAndCrossAttentions,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def forward(
#         self,
#         input_ids=None,
#         past_key_values=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
 
#         # input_ids与inputs_embeds只能输入一个，有input_ids变只需将input_ids输入嵌入层即可变为类似inputs_embeds的张量,
#         # 有inputs_embeds变不需要input_ids
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
 
#         # 下方是确保输入的input_ids、token_type_ids、position_ids等张量的形状为正确的样式:
#         # <1> 若为模型第一次迭代, 则此时input_ids、token_type_ids、position_ids等张量的正确形状为 (batch_size, seq_len),
#         # <2> 若为模型第二次及之后的迭代, 则此时input_ids、token_type_ids、position_ids等张量的正确形状为 (batch_size, 1).
#         # 最后, 将输入的input_ids、token_type_ids、position_ids等张量的形状保存到input_shape中.
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#             input_ids = input_ids.view(-1, input_shape[-1])
#             batch_size = input_ids.shape[0]
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#             batch_size = inputs_embeds.shape[0]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")
 
#         if token_type_ids is not None:
#             token_type_ids = token_type_ids.view(-1, input_shape[-1])
#         if position_ids is not None:
#             position_ids = position_ids.view(-1, input_shape[-1])
 
#         if past_key_values is None:
#             past_length = 0
#             # 若此时为GPT2模型第一次迭代, 则不存在上一次迭代返回的past_key_values列表(包含12个present的列表,
#             # 也就是代码中的presents列表), 则此时past_key_values列表为一个包含12个None值的列表.
#             past_key_values = [None] * len(self.h)
#         else:
#             past_length = past_key_values[0][0].size(-2)
#         if position_ids is None:
#             device = input_ids.device if input_ids is not None else inputs_embeds.device
#             '''<1> GPT2Model第一次迭代时输入GPT2Model的forward()函数中的past_key_values参数为None, 此时past_length为0, 
#               input_shape[-1] + past_length就等于第一次迭代时输入的文本编码(input_ids)的seq_len维度本身, 
#               此时创建的position_ids张量形状为(batch_size, seq_len).
#               <2> 若为GPT2Mode第二次及之后的迭代时, 此时past_length为上一次迭代时记录保存下来的past_key_values中
#               张量的seq_len维度, 而input_shape[-1] + past_length则等于seq_len + 1, 因为在第二次及之后的迭代中,
#               输入的文本编码(input_ids)的seq_len维度本身为1,即第二次及之后的迭代中每次只输入一个字的文本编码,
#               此时创建的position_ids张量形状为(batch_size, 1).'''
#             position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
#             position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
 
#         # Attention mask.
#         # attention_mask张量为注意力遮罩张量, 其让填充特殊符[PAD]处的注意力分数极小,其embedding嵌入值
#         # 基本不会在多头注意力聚合操作中被获取到.
#         if attention_mask is not None:
#             assert batch_size > 0, "batch_size has to be defined and > 0"
#             attention_mask = attention_mask.view(batch_size, -1)
#             # We create a 3D attention mask from a 2D tensor mask.
#             # Sizes are [batch_size, 1, 1, to_seq_length]
#             # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
#             # this attention mask is more simple than the triangular masking of causal attention
#             # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
#             attention_mask = attention_mask[:, None, None, :]
 
#             # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
#             # masked positions, this operation will create a tensor which is 0.0 for
#             # positions we want to attend and -10000.0 for masked positions.
#             # Since we are adding it to the raw scores before the softmax, this is
#             # effectively the same as removing these entirely.
#             attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
#             attention_mask = (1.0 - attention_mask) * -10000.0
 
#         # If a 2D ou 3D attention mask is provided for the cross-attention
#         # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length],
#         # 若此时有从编码器encoder中传入的编码器隐藏状态encoder_hidden_states, 则获取编码器隐藏状态encoder_hidden_states
#         # 的形状(encoder_batch_size, encoder_sequence_length), 同时定义编码器隐藏状态对应的attention_mask张量(即encoder_attention_mask).
#         if self.config.add_cross_attention and encoder_hidden_states is not None:
#             encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#             if encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
#             encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#         else:
#             encoder_attention_mask = None
 
#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # head_mask has shape n_layer x batch x n_heads x N x N
#         # prune_heads()可结合 https://github.com/huggingface/transformers/issues/850 理解.
#         head_mask = self.get_head_mask(head_mask, self.config.n_layer)
 
#         # 将input_ids、token_type_ids、position_ids等张量输入嵌入层self.wte()、 self.wpe()中之后获取其嵌入形式张量
#         # inputs_embeds、position_embeds与token_type_embeds.
#         if inputs_embeds is None:
#             inputs_embeds = self.wte(input_ids)
#         position_embeds = self.wpe(position_ids)
#         hidden_states = inputs_embeds + position_embeds
 
#         if token_type_ids is not None:
#             token_type_embeds = self.wte(token_type_ids)
#             hidden_states = hidden_states + token_type_embeds
 
#         '''<1> GPT2Model第一次迭代时输入GPT2Model的forward()函数中的past_key_values参数为None, 此时past_length为0, 
#               此时hidden_states张量形状为(batch_size, sel_len, n_embd)，config的GPT2Config()类中n_emb默认为768.
#           <2> 若为GPT2Mode第二次及之后的迭代时, 此时past_length为上一次迭代时记录保存下来的past_key_values中
#               张量的seq_len维度, 而input_shape[-1] + past_length则等于seq_len + 1, 因为在第二次及之后的迭代中,
#               输入的文本编码(input_ids)的seq_len维度本身为1,即第二次及之后的迭代中每次只输入一个字的文本编码,
#               此时hidden_states张量形状为(batch_size, 1, n_embd)，config的GPT2Config()类中n_emb默认为768.'''
#         hidden_states = self.drop(hidden_states)
 
#         output_shape = input_shape + (hidden_states.size(-1),)
 
#         # config对应的GPT2Config()类中的use_cache默认为True.
#         presents = () if use_cache else None
#         all_self_attentions = () if output_attentions else None
#         all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
#         all_hidden_states = () if output_hidden_states else None
 
#         for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
#             '''此处past_key_values元组中一共有12个元素(layer_past), 分别对应GPT2模型中的12层Transformer_Block,
#             每一个layer_past都为模型上一次迭代中每个Transformer_Block保留下来的present张量, 而每个present张量保存着
#             Transformer_Block中Attention模块将本次迭代的key张量与上一次迭代中的past_key张量(layer_past[0])合并、
#             将本次迭代的value张量与上一次迭代中的past_value张量(layer_past[1])合并所得的新的key张量与value张量,
#             之后保存着本次迭代中12层Transformer_Block每一层中返回的present张量的presents元组, 便会被作为下一次迭代中
#             的past_key_values元组输入进下一次迭代的GPT2模型中。
#             新的key张量与value张量详细解析如下：'''
 
#             '''第一次迭代时query、key、value张量的seq_len维度处的维度数就为seq_len而不是1, 第二次之后seq_len维度的维度数皆为1.'''
 
#             '''<1> 本次迭代中新的key张量
#             此时需要通过layer_past[0].transpose(-2, -1)操作将past_key张量的形状变为(batch_size, num_head, head_features, sql_len),
#             而此时key张量的形状为(batch_size, num_head, head_features, 1), 这样在下方就方便将past_key张量与key张量在最后
#             一个维度(dim=-1)处进行合并, 这样就将当前token的key部分加入了past_key的seq_len部分, 以方便模型在后面预测新的token,
#             此时新的key张量的形状为: (batch_size, num_head, head_features, sql_len+1), new_seq_len为sql_len+1。
#              <2>  本次迭代中新的value张量
#             而此时past_value(layer_past[1])不用变形, 其形状为(batch_size, num_head, sql_len, head_features), 
#             而此时value张量的形状为(batch_size, num_head, 1, head_features), 这样在下方就方便将past_value张量与value张量
#             在倒数第二个维度(dim=-2)处进行合并, 这样就将当前token的value部分加入了past_value的seq_len部分, 
#             以方便模型在后面预测新的token,
#             此时新的value张量的形状为: (batch_size, num_head, sql_len+1, head_features), new_seq_len为sql_len+1。'''
 
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)
 
#             if getattr(self.config, "gradient_checkpointing", False):
 
#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         # checkpointing only works with tuple returns, not with lists
#                         return tuple(output for output in module(*inputs, use_cache, output_attentions))
 
#                     return custom_forward
 
#                 outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(block),
#                     hidden_states,
#                     layer_past,
#                     attention_mask,
#                     head_mask[i],
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                 )
#             else:
#                 # 此时返回的outputs列表中的元素为：
#                 # <1> 第一个值为多头注意力聚合操作结果张量hidden_states输入前馈MLP层与残差连接之后得到的hidden_states张量,
#                 #     形状为(batch_size, 1, n_state), all_head_size=n_state=nx=n_embd=768.
#                 # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
#                 #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
#                 # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w.
#                 # <4> 若此时进行了Cross Attention计算, 则第四个值为'交叉多头注意力计算结果列表cross_attn_outputs'中的
#                 #     交叉注意力分数张量cross_attention, 其形状为(batch_size, num_head, 1, enc_seq_len).
#                 outputs = block(
#                     hidden_states,
#                     layer_past=layer_past,
#                     attention_mask=attention_mask,
#                     head_mask=head_mask[i],
#                     encoder_hidden_states=encoder_hidden_states,
#                     encoder_attention_mask=encoder_attention_mask,
#                     use_cache=use_cache,
#                     output_attentions=output_attentions,
#                 )
 
#             hidden_states, present = outputs[:2]
#             if use_cache is True:
#                 presents = presents + (present,)
 
#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (outputs[2],)
#                 if self.config.add_cross_attention:
#                     all_cross_attentions = all_cross_attentions + (outputs[3],)
 
#         # 将GPT2模型中12层Block模块计算后得到的最终hidden_states张量再输入进LayerNormalization层中进行计算.
#         hidden_states = self.ln_f(hidden_states)
 
#         hidden_states = hidden_states.view(*output_shape)
#         # Add last hidden state, 即将上方最后一层Block()循环结束之后得到的结果隐藏状态张量hidden_states
#         # 也添加入元组all_hidden_states中.
#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)
 
#         # 此时返回的元素为：
#         # <1> 第一个值为GPT2模型中经过12层Block模块计算后得到的最终hidden_states张量,
#         #     形状为(batch_size, 1, n_state), all_head_size=n_state=nx=n_embd=768.
#         # <2> 第二个值为GPT2模型中12层Block模块计算后得到的存储12个present张量的presents元组, 每一个present张量存储着
#         #     past_key张量与这次迭代的key张量合并后的新key张量, 以及past_value张量与这次迭代的value张量合并后的新value张量,
#         #     一个present张量形状为(2, batch_size, num_head, sql_len+1, head_features).
#         # <3> 若output_hidden_states为True, 则第三个值为GPT2模型中12层Block模块计算后得到的存储12个隐藏状态张量hidden_states
#         #     的all_hidden_states元组.
#         # <4> 若output_attentions为True, 则第四个值为GPT2模型中12层Block模块计算后得到的存储12个注意力分数张量w
#         #     的all_self_attentions元组.
#         # <5> 若此时进行了Cross Attention计算, 则第五个值为GPT2模型中12层Block模块计算后得到的存储12个交叉注意力分数张量
#         #     cross_attention的all_cross_attentions元组,
#         #     其中每个交叉注意力分数张量cross_attention形状为(batch_size, num_head, 1, enc_seq_len).
#         if not return_dict:
#             return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
 
#         return BaseModelOutputWithPastAndCrossAttentions(
#             last_hidden_state=hidden_states,
#             past_key_values=presents,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#             cross_attentions=all_cross_attentions,
#         )




# class Block(nn.Module):
#     def __init__(self, n_ctx, config, scale=False):
#         super().__init__()
#         # config对应的GPT2Config()类中, n_embd属性默认为768, 因此此处hidden_size即为768.
#         hidden_size = config.n_embd
#         # config对应的GPT2Config()类中, n_inner属性默认为None, 因此此处inner_dim一般都为4 * hidden_size.
#         inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

#         self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
#         # 此处n_ctx即等于config对应的GPT2Config()类中的n_ctx属性, 其值为1024.
#         self.attn = Attention(hidden_size, n_ctx, config, scale)
#         self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

#         if config.add_cross_attention:
#             self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
#             self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
#         self.mlp = MLP(inner_dim, config)

#     def forward(
#         self,
#         hidden_states,
#         layer_past=None,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         use_cache=False,
#         output_attentions=False,
#     ):
        
#         '''
#         <1> 此时的隐藏状态hidden_states的形状为 (batch_size, 1, nx), 此时nx = n_state = n_embed = all_head_size = 768，
#             即此时隐藏状态hidden_states的形状为(batch_size, 1, 768)。
#         <2> 此时layer_past为一个存储着past_key张量与past_value张量的大张量, 其
#              形状为(2, batch_size, num_head, sql_len, head_features).
#         <3> attention_mask张量为注意力遮罩张量, 其让填充特殊符[PAD]处的注意力分数极小,
#              其embedding嵌入值基本不会在多头注意力聚合操作中被获取到.
#         '''

#         # 将此时输入的隐藏状态hidden_states先输入进LayerNormalization层进行层标准化计算后,
#         # 再将标准化结果输入进'多头注意力计算层self.attn()'中进行多头注意力聚合操作计算.
#         # 此时返回的attn_outputs列表中:
#         # <1> 第一个值为多头注意力聚合操作结果张量a, 形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
#         # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
#         #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
#         # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w.
#         attn_outputs = self.attn(
#             self.ln_1(hidden_states),
#             layer_past=layer_past,
#             attention_mask=attention_mask,
#             head_mask=head_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#         )

#         # 此时的attn_output张量为返回的attn_outputs列表中第一个值:
#         # 多头注意力聚合操作结果张量a, 形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
#         attn_output = attn_outputs[0]  # output_attn列表: a, present, (attentions)
#         outputs = attn_outputs[1:]

#         # residual connection, 进行残差连接.
#         # 此时attn_output张量形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
#         # hidden_states的形状为(batch_size, 1, 768).
#         hidden_states = attn_output + hidden_states


#         if encoder_hidden_states is not None:
#             # add one self-attention block for cross-attention
#             assert hasattr(
#                 self, "crossattention"
#             ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"


#             '''此时self.crossattention()的Cross_Attention运算过程与self.attn()的Attention运算过程几乎相同, 其不同点在于：

#             <1> self.attn()的Attention运算是将LayerNormalization之后的hidden_states通过'self.c_attn = Conv1D(3 * n_state, nx)
#             (第165行代码)'将hidden_states的形状由(batch_size,1, 768)投影为(batch_size,1, 3 * 768), 再将投影后的hidden_states
#             在第三维度(dim=2)上拆分为三份分别赋为query、key、value, 其形状都为(batch_size, 1, 768)；
# 			此时n_state = nx = num_head*head_features = 768.
			
#             之后经过split_heads()函数拆分注意力头且key、value张量分别与past_key、past_value张量合并之后:
#             query张量的形状变为(batch_size, num_head, 1, head_features),
#             key张量的形状变为(batch_size, num_head, head_features, sql_len+1),
#             value张量的形状变为(batch_size, num_head, sql_len+1, head_features).

#             <2> self.crossattention()的Cross_Attention运算过程则是将LayerNormalization之后的hidden_states通过
#             'self.q_attn = Conv1D(n_state, nx)(第163行代码)'将hidden_states的形状由(batch_size,1, 768)投影为(batch_size,1, 768),
#             将此投影之后的hidden_states赋值作为query张量；
#             再将此时从编码器(encoder)中传过来的编码器隐藏状态encoder_hidden_states通过'self.c_attn = Conv1D(2 * n_state, nx)
#             (第162行代码)'将encoder_hidden_states的形状由(batch_size, enc_seq_len, 768)投影为(batch_size, enc_seq_len, 2 * 768),
#             将投影后的encoder_hidden_states在在第三维度(dim=2)上拆分为两份分别赋为key、value,
#             其形状都为(batch_size, enc_seq_len, 768)； 此时n_state = nx = num_head*head_features = 768.
            
#             之后经过split_heads()函数拆分注意力头之后:
#             query张量的形状变为(batch_size, num_head, 1, head_features),
#             key张量的形状变为(batch_size, num_head, head_features, enc_seq_len),
#             value张量的形状变为(batch_size, num_head, enc_seq_len, head_features).
#             此时计算出的cross_attention张量形状为(batch_size, num_head, 1, enc_seq_len).'''

#             # 此时将上方的隐藏状态hidden_states(Attention运算结果+Attention运算前的hidden_states)先输入进LayerNormalization
#             # 层进行层标准化计算后, 再将标准化结果输入进'交叉多头注意力计算层self.crossattention()'中与编码器传入的隐藏状态
#             # encoder_hidden_states进行交叉多头注意力聚合操作计算.
#             # 此时返回的cross_attn_outputs列表中:
#             # <1> 第一个值为与编码器传入的隐藏状态encoder_hidden_states进行交叉多头注意力聚合操作的结果张量a,
#             #     形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768。
#             # <2> 第二个值仍为present张量, 但由于此时是做'交叉多头注意力计算self.crossattention()',此时输入进self.crossattention()
#             #     函数的参数中不包含layer_past(来自past_key_values列表)的past_key与past_value张量, 因此此时的present为(None,),
#             #     详细代码可见本脚本代码357行, 因此此处用不到'交叉多头注意力计算结果列表cross_attn_outputs'中的present,
#             #     将其舍弃(代码第528行)。
#             # <3> 若output_attentions为True, 则第三个值为: 交叉注意力分数张量w, 即cross attentions,
#             #      cross_attention张量形状为(batch_size, num_head, 1, enc_seq_len).
#             cross_attn_outputs = self.crossattention(
#                 self.ln_cross_attn(hidden_states),
#                 attention_mask=attention_mask,
#                 head_mask=head_mask,
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=encoder_attention_mask,
#                 output_attentions=output_attentions,
#             )
#             attn_output = cross_attn_outputs[0]
#             # residual connection
#             hidden_states = hidden_states + attn_output
#             # cross_attn_outputs[2:] add cross attentions if we output attention weights,
#             # 即将'交叉多头注意力计算结果列表cross_attn_outputs'中的交叉注意力分数张量cross_attention保存为此时的
#             # outputs列表中的最后一个元素.
#             outputs = outputs + cross_attn_outputs[2:]


#         feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
#         # residual connection
#         hidden_states = hidden_states + feed_forward_hidden_states

#         outputs = [hidden_states] + outputs

#         # 此时返回的outputs列表中的元素为：
#         # <1> 第一个值为多头注意力聚合操作结果张量hidden_states输入前馈MLP层与残差连接之后得到的最终hidden_states张量,
#         #     形状为(batch_size, 1, n_state), all_head_size=n_state=nx=n_embd=768.
#         # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
#         #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
#         # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w.
#         # <4> 若此时进行了Cross Attention计算, 则第四个值为'交叉多头注意力计算结果列表cross_attn_outputs'中的
#         #     交叉注意力分数张量cross_attention, 其形状为(batch_size, num_head, 1, enc_seq_len).
#         return outputs  # hidden_states, present, (attentions, cross_attentions)




# class Attention(nn.Module):
#     def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
#         super().__init__()

#         n_state = nx  # in Attention: n_state=768 (nx=n_embd)
#         # [switch nx => n_state from Block to Attention to keep identical to TF implem]
#         # 利用断言函数判断此时隐藏状态的维度数n_state除以注意力头数config.n_head之后是否能整除.
#         assert n_state % config.n_head == 0

#         # 下方的self.register_buffer()函数的操作相当于创建了两个Attention类中的self属性, 即为self.bias属性
#         # 与self.masked_bias属性；
#         # 其中self.bias属性为一个下三角矩阵(对角线下元素全为1, 对角线上元素全为0), 其形状为(1, 1, n_ctx, n_ctx),
#         # 也即形状相当于(1, 1, 1024, 1024)；
#         # 而self.masked_bias属性则为一个极大的负数-1e4；
#         self.register_buffer(
#             "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
#         )
#         self.register_buffer("masked_bias", torch.tensor(-1e4))


#         self.n_head = config.n_head
#         self.split_size = n_state
#         self.scale = scale

#         self.is_cross_attention = is_cross_attention
#         if self.is_cross_attention:
#             # self.c_attn = Conv1D(2 * n_state, nx)相当于全连接层, 其将输入张量的最后一个维度的维度数由nx(768)投影为
#             # 2 * n_state(2*768), 此时n_state = nx = num_head*head_features = 768.
#             self.c_attn = Conv1D(2 * n_state, nx)

#             # self.q_attn = Conv1D(n_state, nx)相当于全连接层, 其将输入张量的最后一个维度的维度数由nx(768)投影为
#             # n_state(768), 此时n_state = nx = num_head*head_features = 768.
#             self.q_attn = Conv1D(n_state, nx)

#         else:
#             # self.c_attn = Conv1D(3 * n_state, nx)相当于全连接层, 其将输入张量的最后一个维度的维度数由nx(768)投影为
#             # 2 * n_state(2*768), 此时n_state = nx = num_head*head_features = 768.
#             self.c_attn = Conv1D(3 * n_state, nx)

#         # 此处self.c_proj()为Conv1D(n_state, nx)函数(all_head_size=n_state=nx=768), 相当于一个全连接层的作用,
#         # 其将此时的多头注意力聚合操作结果张量a的最后一个维度all_head_size由n_state(768)的维度数投影为nx(768)的维度数.
#         self.c_proj = Conv1D(n_state, nx)
#         self.attn_dropout = nn.Dropout(config.attn_pdrop)
#         self.resid_dropout = nn.Dropout(config.resid_pdrop)
#         self.pruned_heads = set()


#     # prune_heads()可结合 https://github.com/huggingface/transformers/issues/850 理解.
#     def prune_heads(self, heads):
#         if len(heads) == 0:
#             return
#         heads, index = find_pruneable_heads_and_indices(
#             heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
#         )
#         index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

#         # Prune conv1d layers
#         self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
#         self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

#         # Update hyper params
#         self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
#         self.n_head = self.n_head - len(heads)
#         self.pruned_heads = self.pruned_heads.union(heads)


#     def merge_heads(self, x):
#         # 此时x为: 利用计算得到的注意力分数张量对value张量进行注意力聚合后得到的注意力结果张量.
#         # x的形状为(batch_size, num_head, sql_len, head_features).

#         # 此时先将注意力结果张量x的形状变为(batch_size, sql_len, num_head, head_features)
#         x = x.permute(0, 2, 1, 3).contiguous()
#         # new_x_shape为(batch_size, sql_len, num_head*head_features) =》(batch_size, sql_len, all_head_size)
#         new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)

#         # 此时将注意力结果张量x的注意力头维度num_head与注意力特征维度head_features进行合并变为all_head_size维度,
#         # 注意力结果张量x的形状变为(batch_size, sql_len, all_head_size).
#         return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states， (batch_size, sql_len, all_head_size).


#     def split_heads(self, x, k=False):
#         # 此时new_x_shape为: (batch_size, sql_len, num_head, head_features)
#         new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
#         # 将输入的张量x(可能为query、key、value张量)变形为: (batch_size, sql_len, num_head, head_features).
#         x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states

#         # 若此时输入的张量为key张量,则需要将key张量再变形为(batch_size, num_head, head_features, sql_len).
#         # 因为此时key张量需要以[query * key]的形式与query张量做内积运算, 因此key张量需要将head_features变换到第三维度,
#         # 将sql_len变换到第四维度,这样[query * key]内积运算之后的注意力分数张量的形状才能符合(batch_size, num_head, sql_len, sql_len).
#         if k:
#             return x.permute(0, 2, 3, 1)  # (batch_size, num_head, head_features, sql_len)

#         # 若此时输入的张量为query张量或value张量, 则将张量维度再变换为(batch_size, num_head, sql_len, head_features)即可,
#         # 即将sql_len与num_head调换维度.
#         else:
#             return x.permute(0, 2, 1, 3)  # (batch_size, num_head, sql_len, head_features)


#     def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        
#         '''
#         此时query张量形状为: (batch_size, num_head, 1, head_features)
#         key张量的形状为: (batch_size, num_head, head_features, sql_len+1)
#         value张量的形状为: (batch_size, num_head, sql_len+1, head_features)

#         此时key张量以[query * key]的形式与query张量做内积运算, key张量已在split_heads()操作与past_key合并操作中
#         提前将head_features变换到第三维度, 将sql_len+1变换到第四维度,这样[query * key]内积运算之后的注意力分数张量w的
#         形状才能符合(batch_size, num_head, 1, sql_len+1).
#         '''
#         w = torch.matmul(q, k)  # 注意力分数张量w: (batch_size, num_head, 1, sql_len+1)

#         # 对注意力分数张量w中的值进行缩放(scaled), 缩放的除数为注意力头特征数head_features的开方值.
#         if self.scale:
#             w = w / (float(v.size(-1)) ** 0.5)

#         # 此时nd与ns两个维度相当于1与seq_len+1
#         nd, ns = w.size(-2), w.size(-1)

#         # 此处的操作为利用torch.where(condition, x, y)函数,将注意力分数张量w在mask.bool()条件张量为True(1)的相同位置的值
#         # 保留为w中的原值, 将在mask.bool()条件张量为True(0)的相同位置的值变为self.masked_bias(-1e4)的值.
#         '''<1> GPT2Model第一次迭代时输入GPT2Model的forward()函数中的past_key_values参数为None, 此时nd与ns维度才会相等, 
#         在nd与ns维度相等的情况下此操作的结果等价于让注意力分数张量w与attention_mask张量相加的结果。
#         <2> 若为GPT2Mode第二次及之后的迭代时, nd与ns两个维度相当于1与seq_len+1, 此时对self.bias进行切片操作时, 
#         ns - nd等于seq_len+1 - 1即结果为seq_len, 即此时切片操作相当于self.bias[:, :, seq_len : seq_len+1, :seq_len+1],
#         此操作的意义在于对此次迭代中, 最新的token的注意力分数上添加GPT2中的下三角形式的注意力遮罩.'''
#         if not self.is_cross_attention:
#             # if only "normal" attention layer implements causal mask
#             # 此时self.bias属性为一个下三角矩阵(对角线下元素全为1, 对角线上元素全为0), 其形状为(1, 1, n_ctx, n_ctx),
#             # 也即形状相当于(1, 1, 1024, 1024)；但此处对self.bias进行切片操作时, ns - nd等于seq_len+1 - 1即结果为seq_len,
#             # 即此时切片操作相当于self.bias[:, :, seq_len : seq_len+1, :seq_len+1]。
#             '''此时mask张量(经过大张量self.bias切片获得)的形状为(1, 1, 1, seq_len + 1).'''
#             mask = self.bias[:, :, ns - nd: ns, :ns]
#             '''此操作的意义在于对此次迭代中, 最新的token的注意力分数上添加GPT2中的下三角形式注意力遮罩.'''
#             w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

#         # 让注意力分数张量w与attention_mask张量相加, 以达到让填充特殊符[PAD]处的注意力分数为一个很大的负值的目的,这样在下面将
#         # 注意力分数张量w输入Softmax()层计算之后, 填充特殊符[PAD]处的注意力分数将会变为无限接近0的数, 以此让填充特殊符[PAD]
#         # 处的注意力分数极小, 其embedding嵌入值基本不会在多头注意力聚合操作中被获取到.
#         if attention_mask is not None:
#             # Apply the attention mask
#             w = w + attention_mask

#         # 注意力分数张量w: (batch_size, num_head, 1, sql_len+1).
#         # 将注意力分数张量w输入进Softmax()层中进行归一化计算, 计算得出最终的注意力分数,
#         # 再将注意力分数张量w输入进Dropout层self.attn_dropout()中进行正则化操作, 防止过拟合.
#         w = nn.Softmax(dim=-1)(w)
#         w = self.attn_dropout(w)

#         # Mask heads if we want to, 对注意力头num_head维度的mask操作.
#         if head_mask is not None:
#             w = w * head_mask

#         # 多头注意力聚合操作: 注意力分数张量w与value张量进行内积
#         # 注意力分数张量w形状: (batch_size, num_head, 1, sql_len+1)
#         # value张量形状: (batch_size, num_head, sql_len+1, head_features)
#         # 多头注意力聚合操作结果张量形状: (batch_size, num_head, 1, head_features), head_features=768.
#         outputs = [torch.matmul(w, v)]
#         # 若同时返回注意力分数张量w, 则将w张量添加入outputs列表中.
#         if output_attentions:
#             outputs.append(w)

#         return outputs


#     def forward(
#         self,
#         hidden_states,
#         layer_past=None,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         use_cache=False,
#         output_attentions=False,
#     ):
#         # <1> 此时的隐藏状态hidden_states的形状为 (batch_size, 1, nx), 此时nx = n_state = n_embed = head_features = 768，
#         #     即此时隐藏状态hidden_states的形状为(batch_size, 1, 768)。
#         # <2> 此时layer_past为一个存储着past_key张量与past_value张量的大张量, 其
#         #     形状为(2, batch_size, num_head, sql_len, head_features).
#         # <3> attention_mask张量为注意力遮罩张量, 其让填充特殊符[PAD]处的注意力分数极小,
#         #     其embedding嵌入值基本不会在多头注意力聚合操作中被获取到.

#         if encoder_hidden_states is not None:
#             assert hasattr(
#                 self, "q_attn"
#             ), "If class is used as cross attention, the weights `q_attn` have to be defined. " \
#                "Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."

#             '''self.crossattention()的Cross_Attention运算过程则是将LayerNormalization之后的hidden_states通过
#             'self.q_attn = Conv1D(n_state, nx)(第168行代码)'将hidden_states的形状由(batch_size,1, 768)投影为(batch_size,1, 768),
#             将此投影之后的hidden_states赋值作为query张量；
#             再将此时从编码器(encoder)中传过来的编码器隐藏状态encoder_hidden_states通过'self.c_attn = Conv1D(2 * n_state, nx)
#             (第164行代码)'将encoder_hidden_states的形状由(batch_size, enc_seq_len, 768)投影为(batch_size, enc_seq_len, 2 * 768),
#             将投影后的encoder_hidden_states在在第三维度(dim=2)上拆分为两份分别赋为key、value,
#             其形状都为(batch_size, enc_seq_len, 768)；  此时n_state = nx = num_head*head_features = 768.
            
#             之后经过split_heads()函数拆分注意力头之后:
#             query张量的形状变为(batch_size, num_head, 1, head_features),
#             key张量的形状变为(batch_size, num_head, head_features, enc_seq_len),
#             value张量的形状变为(batch_size, num_head, enc_seq_len, head_features).
            
#             此时计算出的cross_attention张量形状为(batch_size, num_head, 1, enc_seq_len).'''

#             query = self.q_attn(hidden_states)
#             key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
#             attention_mask = encoder_attention_mask

#         else:
#             '''此时隐藏状态hidden_states的形状为(batch_size, 1, 768), 将其输入进全连接层self.c_attn中后,
#             其Conv1D(3 * n_state, nx)操作(nx=n_state=768)便会将hidden_states的第三维度数由 768维 投影为 3 * 768维,
#             此时的hidden_states张量的形状为(batch_size, 1, 3 * 768), 最后将hidden_states张量在第三个维度(维度数3 * 768)上
#             切分为三块, 将这切分出的三块各当成query, key, value张量, 则每个张量的形状都为(batch_size, 1, 768).
#             此时n_state = nx = num_head*head_features = 768.
            
#             之后经过split_heads()函数拆分注意力头且key、value张量分别与past_key、past_value张量合并之后:
#             query张量的形状变为(batch_size, num_head, 1, head_features),
#             key张量的形状变为(batch_size, num_head, head_features, sql_len+1),
#             value张量的形状变为(batch_size, num_head, sql_len+1, head_features).'''
#             query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)


#         '''第一次迭代时query、key、value张量的seq_len维度处的维度数就为seq_len而不是1, 第二次之后seq_len维度的维度数皆为1.'''
#         # 此时经过'注意力头拆分函数split_heads()'之后的query、key、value三个张量的形状分别为:
#         # query: (batch_size, num_head, 1, head_features)
#         # key: (batch_size, num_head, head_features, 1)
#         # value: (batch_size, num_head, 1, head_features)
#         query = self.split_heads(query)
#         key = self.split_heads(key, k=True)
#         value = self.split_heads(value)

#         if layer_past is not None:
#             '''第一次迭代时query、key、value张量的seq_len维度处的维度数就为seq_len而不是1, 第二次之后seq_len维度的维度数皆为1.'''
#             '''<1> 本次迭代中新的key张量
#             此时需要通过layer_past[0].transpose(-2, -1)操作将past_key张量的形状变为(batch_size, num_head, head_features, sql_len),
#             而此时key张量的形状为(batch_size, num_head, head_features, 1), 这样在下方就方便将past_key张量与key张量在最后
#             一个维度(dim=-1)处进行合并, 这样就将当前token的key部分加入了past_key的seq_len中, 以方便模型在后面预测新的token,
#             此时新的key张量的形状为: (batch_size, num_head, head_features, sql_len+1), new_seq_len为sql_len+1。
#              <2> 本次迭代中新的value张量
#             而此时past_value不用变形, 其形状为(batch_size, num_head, sql_len, head_features), 而此时value张量的形状为
#             (batch_size, num_head, 1, head_features), 这样在下方就方便将past_value张量与value张量在倒数第二个
#             维度(dim=-2)处进行合并, 这样就将当前token的value部分加入了past_value的seq_len中, 以方便模型在后面预测新的token,
#             此时新的value张量的形状为: (batch_size, num_head, sql_len+1, head_features), new_seq_len为sql_len+1。
#            '''
#             past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
#             key = torch.cat((past_key, key), dim=-1)
#             value = torch.cat((past_value, value), dim=-2)

#         # config对应的GPT2Config()类中的use_cache默认为True.但此时若为Cross_Attention运算过程, 则此时不会指定use_cache,
#         # 而此时use_cache属性即为False(因为Attention类中use_cache属性默认为False, 除非指定config对应的GPT2Config()类
#         # 中的use_cache属性其才会为True).
#         if use_cache is True:
#             # 若use_cache为True, 此时将key张量的最后一个维度与倒数第二个维度互换再与value张量进行stack合并,
#             # 此时key.transpose(-2, -1)的形状为(batch_size, num_head, sql_len+1, head_features),
#             # 此时torch.stack()操作后的present张量形状为(2, batch_size, num_head, sql_len+1, head_features)。
#             '''present张量形状: (2, batch_size, num_head, sql_len+1, head_features),
#             即present张量是用来存储此次迭代中的key张量与上一次迭代中的past_key张量(layer_past[0])合并、
#             本次迭代的value张量与上一次迭代中的past_value张量(layer_past[1])合并后所得的新的key张量与value张量的.'''
#             present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
#         else:
#             present = (None,)


#         '''此时query张量形状为: (batch_size, num_head, 1, head_features)
#         key张量的形状为: (batch_size, num_head, head_features, sql_len+1)
#         value张量的形状为: (batch_size, num_head, sql_len+1, head_features)'''
#         # 若output_attentions为True, 则self._attn()函数返回的attn_outputs列表中的第二个值为注意力分数张量w.
#         attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)


#         # 此时self._attn()函数返回的attn_outputs列表中的第一个元素为多头注意力聚合操作结果张量a,
#         # a张量的形状为(batch_size, num_head, 1, head_features);
#         # 若output_attentions为True, 则此时self._attn()函数返回的attn_outputs列表中的第二个元素为
#         # 注意力分数张量w, 其形状为(batch_size, num_head, 1, seq_len + 1).
#         a = attn_outputs[0]

#         '''此时经过'多头注意力头合并函数self.merge_heads()'后的多头注意力聚合操作结果张量a的形状
#         变为(batch_size, 1, all_head_size), 其中 all_head_size 等于 num_head * head_features, head_features=768.
#         all_head_size维度的维度数为768,等于n_state,也等于nx, 即all_head_size=n_state=nx=768.'''
#         a = self.merge_heads(a)

#         # 此处self.c_proj()为Conv1D(n_state, nx)函数(all_head_size=n_state=nx=768), 相当于一个全连接层的作用,
#         # 其将此时的多头注意力聚合操作结果张量a的最后一个维度all_head_size由n_state(768)的维度数投影为nx(768)的维度数.
#         a = self.c_proj(a)
#         a = self.resid_dropout(a)  # 残差dropout层进行正则化操作, 防止过拟合.

#         # 此时多头注意力聚合操作结果张量a的形状为(batch_size, 1, all_head_size),
#         # 其中 all_head_size 等于 num_head * head_features；all_head_size维度的维度数为768,
#         # 等于n_state,也等于nx, 即all_head_size=n_state=nx=n_embed=768.
#         outputs = [a, present] + attn_outputs[1:]

#         # 此时返回的outputs列表中:
#         # <1> 第一个值为多头注意力聚合操作结果张量a, 形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
#         # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
#         #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
#         # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w,
#         #     其形状为(batch_size, num_head, 1, seq_len + 1).
#         return outputs  # a, present, (attentions)


