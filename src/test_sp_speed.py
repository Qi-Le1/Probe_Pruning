import torch
import torch.nn as nn
import time
import torch
import time
import torch
import time
import numpy as np
# Ensure PyTorch is using CUDA
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print("Using CUDA:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device('cpu')
#     print("CUDA is not available. Using CPU instead.")

# def low_rank_approximation(U, S, V, k):
#     S_k = torch.diag(S[:k]).to(device)
#     U_k = U[:, :k].to(device)
#     V_k = V[:k, :].to(device)
#     # A_k = U_k @ S_k @ V_k.T
#     return U_k, S_k, V_k

# def measure_time(input_tensor, A):
#     if device == torch.device('cpu'):
#         result = torch.matmul(input_tensor.to(device), A.T)
#         print('result original', result)
#         return 0
#     else:
#         torch.cuda.synchronize()  # Wait for CUDA operations to complete
#         start_time = time.perf_counter()
#         # Perform explicit low-rank approximation multiplication here
#         result = torch.matmul(input_tensor.to(device), A.T)
#         print('result original', result)
#         torch.cuda.synchronize()  # Ensure the operation is completed
#         end_time = time.perf_counter()
#     return end_time - start_time

# def measure_svd_time(input_tensor, U_k, S_k, V_k, low_rank_res):
#     if device == torch.device('cpu'):
#         print('S_k', S_k.shape)
#         print('V_k', V_k.shape)
#         # S_k_matrix = torch.diag(S_k)

#         # Assuming V_k represents the right singular vectors, ensure it's transposed appropriately
#         # result = U_k @ S_k_matrix @ V_k.T @ input_tensor.to(device).T
#         a = input_tensor.to(device) @ V_k.T
#         print('a', a.shape)
#         # print('S_k_matrix', S_k_matrix.shape)
#         b = a @ S_k.T
#         c = b @ U_k.T
#         result = c
#         low_rank_res.append(result)
#         print('result', result)
#         return 0
#     else:
#         torch.cuda.synchronize()  # Wait for CUDA operations to complete
#         start_time = time.perf_counter()
#         # Perform explicit low-rank approximation multiplication here
#         # result = input_tensor.to(device) @ V_k @  @ 
#         S_k_matrix = torch.diag(S_k)

#         # Assuming V_k represents the right singular vectors, ensure it's transposed appropriately
#         # result = U_k @ S_k_matrix @ V_k.T @ input_tensor.to(device).T
#         a = V_k @ input_tensor.to(device).T
#         b = S_k_matrix @ a
#         c = U_k @ b
#         result = c
#         # result = torch.matmul(input_tensor.to(device), A.T)
#         # result = input_tensor.to(device) @ 
#         # result = U_k @ S_k @ V_k @ input_tensor.to(device).T
#         print('result', result)
#         torch.cuda.synchronize()  # Ensure the operation is completed
#         end_time = time.perf_counter()
#     return end_time - start_time
# # Create a large matrix A and move it to the chosen device
# A = torch.randn(11008, 4096, device=device)
# if device == torch.device('cpu'):
#     A = torch.randn(100, 40, device=device)

# # Perform SVD on A
# U, S, V = torch.linalg.svd(A, full_matrices=False)
# print("U:", U.shape, "S:", S.shape, "V:", V.shape)

# print("U:", U, "S:", S, "V:", V)
# # for ele in S:
# #     print(ele)
# ks = [64, 128, 640, 4096]  # Low-rank approximations
# if device == torch.device('cpu'):   
#     ks = [30, 40]

# inputs = {
#     "input_10": torch.randn(10, 1024, 4096, device=device),
#     "input_100": torch.randn(100, 1024, 4096, device=device)
# }
# if device == torch.device('cpu'):
#     inputs = { 
#         "input_10": torch.randn(10, 5, 40, device=device),
#         # "input_100": torch.randn(100, 5, 40, device=device)
#     }

# low_rank_res = []
# # Measure and compare execution times using CUDA
# for input_name, input_tensor in inputs.items():
#     print(f"Results for {input_name}:")
#     time_full_matrix = measure_time(input_tensor, A)
#     print(f"Full matrix multiplication on CUDA took {time_full_matrix:.4f} seconds")
    
#     for k in ks:
#         U_k, S_k, V_k = low_rank_approximation(U, S, V, k)
#         # print(f"Low-rank (k={k}) approximation shape:", A_k.shape)
#         time_low_rank = measure_svd_time(input_tensor, U_k, S_k, V_k, low_rank_res)
#         print(f"Low-rank (k={k}) multiplication on CUDA took {time_low_rank:.4f} seconds")

# print('---------------')
# # for i in range(2):
# res = np.sign(low_rank_res[0].numpy()) == np.sign(low_rank_res[1].numpy()).all()
# # print(res)
# res = np.sum(res)
# shape_example = low_rank_res[0].shape  # This is from the previously used tensor1_numpy as a stand-in for low_rank_res[0]
# print(shape_example)
# # Calculate the product of all dimensions in the shape
# product_of_shape = np.prod(shape_example)
# print(res, low_rank_res[0].shape[0], low_rank_res[0].shape[1] )
# print(res/product_of_shape)

d = 4096
m = 11008
k_list = [64, 128, 320, 640]
for k in k_list:
    a = (d * k + k * k + k * m) / (d * m)
    print(a)


# a = (15404*300) / (4096*11008)

# print(a)
# a = (15404*300) / (4096*11008)
# The code snippet provided seems to be a mix of operations that involve tensors and numpy arrays.
# It appears the goal is to compare the sign of elements in two tensors, convert them to numpy arrays,
# see if all corresponding positions have the same sign, and then somehow calculate a fraction based on the size of the tensors.
# Since the operation and context are a bit unclear, especially with the ".all()" possibly being misplaced for the intended logic,
# I'll interpret and correct the code to what seems to be the intended functionality:
# - Compare the signs of elements in two tensors
# - Check if all corresponding elements have the same sign
# - Calculate the fraction of matching signs over the total number of elements

# Assuming low_rank_res is a list of tensors, let's simulate this with numpy arrays for demonstration
# since we don't have the actual tensors and their .numpy() conversion in this environment.

# import numpy as np

# # Example numpy arrays to simulate the tensors after conversion to numpy
# tensor1_numpy = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])
# tensor2_numpy = np.array([[10, -20, 30], [-40, 50, -60], [70, -80, 90]])

# # Calculate the comparison of signs, which simulates np.sign(low_rank_res[0].numpy()) == np.sign(low_rank_res[1].numpy())
# signs_match = np.sign(tensor1_numpy) == np.sign(tensor2_numpy)

# # The corrected logic to calculate the fraction of 'True' values (1s) in the comparison over the total number of elements
# res = signs_match.all()  # This should be True if all match, otherwise False - but seems not what's intended based on the question
# fraction_of_ones = np.mean(signs_match)  # This calculates the fraction of ones

# fraction_of_ones

# Due to environmental constraints, this code is intended for illustration and should be run in a suitable environment.


# # Create a linear layer with dimension 4096 * 11008
# linear_layer = torch.nn.Linear(4096, 11008)

# # Input sample (assuming a batch size of 1 for simplicity)
# input_tensor = torch.rand(1, 4096)

# # Forward pass through the linear layer
# start_time = torch.cuda.Event(enable_timing=True)
# end_time = torch.cuda.Event(enable_timing=True)

# start_time.record()
# output_tensor = linear_layer(input_tensor)
# end_time.record()
# torch.cuda.synchronize()  # Wait for the events to be recorded!
# linear_time = start_time.elapsed_time(end_time)

# # Method 1: Using torch.select to select 80% of the output dimensions
# start_time.record()
# selected_output_1 = output_tensor[:, :int(11008 * 0.8)]
# end_time.record()
# torch.cuda.synchronize()
# select_time = start_time.elapsed_time(end_time)

# # Method 2: Using indexing to select 80% of the output dimensions
# indices = torch.arange(0, int(11008 * 0.8)).long()
# start_time.record()
# selected_output_2 = output_tensor.index_select(1, indices)
# end_time.record()
# torch.cuda.synchronize()
# index_time = start_time.elapsed_time(end_time)

# print("Linear layer forward pass time: {:.4f} ms".format(linear_time))
# print("Select method time: {:.4f} ms".format(select_time))
# print("Index method time: {:.4f} ms".format(index_time))


# class CustomLinearModel(nn.Module):
#     def __init__(self, input_size, output_size, num_layers=20):
#         super(CustomLinearModel, self).__init__()
#         self.layers = nn.ModuleList()
#         for i in range(num_layers):
#             self.layers.append(nn.Linear(input_size, output_size))
#             # Adjust input size for the next layer if needed (for pruning logic)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

# def prune_model(model, prune_percentage=0.1):
#     for i in range(len(model.layers)):
#         if i % 2 == 0:  # Even-indexed layers
#             # Prune 10% of the output dimensions
#             # For simplicity in demonstration, we're adjusting the weight size directly
#             # In practice, you'd use structured pruning methods from torch.nn.utils.prune
#             output_features = model.layers[i].out_features
#             new_out_features = int(output_features * (1 - prune_percentage))
#             model.layers[i] = nn.Linear(model.layers[i].in_features, new_out_features)
#         else:
#             # Adjust the input dimension of the odd layers to match the pruned output of the even layers
#             input_features = model.layers[i].in_features
#             new_in_features = int(input_features * (1 - prune_percentage))
#             model.layers[i] = nn.Linear(new_in_features, model.layers[i].out_features)

# # Parameters
# input_size = 4096
# output_size = 4096
# num_layers = 20
# batch_size = 10
# n_batches = 50

# # Initialize model
# model = CustomLinearModel(input_size, output_size, num_layers)

# # Generate dummy data
# inputs = [torch.randn(batch_size, input_size) for _ in range(n_batches)]

# # Measure runtime without pruning
# start_time = time.time()
# with torch.no_grad():
#     for input in inputs:
#         output = model(input)
# end_time = time.time()
# runtime_without_pruning = end_time - start_time

# # Prune the model
# prune_model(model, 0.1)

# # Measure runtime with pruning
# start_time = time.time()
# with torch.no_grad():
#     for input in inputs:
#         output = model(input)
# end_time = time.time()
# runtime_with_pruning_10 = end_time - start_time

# # Prune the model
# prune_model(model, 0.1)

# # Measure runtime with pruning
# start_time = time.time()
# with torch.no_grad():
#     for input in inputs:
#         output = model(input)
# end_time = time.time()
# runtime_with_pruning_20 = end_time - start_time

# # Prune the model
# prune_model(model, 0.1)

# # Measure runtime with pruning
# start_time = time.time()
# with torch.no_grad():
#     for input in inputs:
#         output = model(input)
# end_time = time.time()
# runtime_with_pruning_30 = end_time - start_time

# # Prune the model
# prune_model(model, 0.1)

# # Measure runtime with pruning
# start_time = time.time()
# with torch.no_grad():
#     for input in inputs:
#         output = model(input)
# end_time = time.time()
# runtime_with_pruning_40 = end_time - start_time

# print("Runtime without pruning: {:.4f} seconds".format(runtime_without_pruning))
# print("Runtime with 10% pruning: {:.4f} seconds".format(runtime_with_pruning_10))
# print("Runtime with 20% pruning: {:.4f} seconds".format(runtime_with_pruning_20))
# print("Runtime with 30% pruning: {:.4f} seconds".format(runtime_with_pruning_30))
# print("Runtime with 40% pruning: {:.4f} seconds".format(runtime_with_pruning_40))

