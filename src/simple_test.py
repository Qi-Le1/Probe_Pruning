
# import torch
# import time
# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("CUDA version:", torch.version.cuda)
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# '''
# PyTorch version: 2.0.1+cu117
# CUDA available: True
# CUDA version: 11.7
# 2.0.1+cu117
# 11.7
# 8906

# System: ubuntu 20.04
# GPU: NVIDIA GeForce RTX 4090
# '''
# import re
# import torch

# # input1 = 'each0.1+0.1'
# # input2 = 'each'
# # input3 = 'each0.1'
# # float_pattern = re.compile(r'\d*\.?\d+')
# # # Find all matches and convert them to floats
# # floats1 = [float(match) for match in float_pattern.findall(input1)]
# # floats2 = [float(match) for match in float_pattern.findall(input2)]
# # floats3 = [float(match) for match in float_pattern.findall(input3)]

# # print(floats1, floats2, floats3)
# import torch

# import matplotlib.pyplot as plt
# import numpy as np

# import matplotlib.pyplot as plt
# import numpy as np

# # Generating example data
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

import torch


empty_tensor = torch.tensor([])
print(empty_tensor.size()) 



# Example l2_norms tensor
# l2_norms = torch.randn(4, 10) * 1000  # 4 batches, 10 sequences each

# # Condition to check
# threshold = 500

# # Finding indices where condition is met
# rows, cols = torch.where(l2_norms > threshold)



# print(rows, cols)\


# batch_indices = torch.tensor([0, 0, 1, 1, ])
# sequence_indices = torch.tensor([2, 4, 5, 7])

# unique_batches, counts = torch.unique_consecutive(batch_indices, return_counts=True)

# # Use counts to split the sequence indices
# result_tensors = torch.split(sequence_indices, counts.tolist())
# print(result_tensors, result_tensors.dtype)
# # result = [[2,4],[5,7]]
# # import torch

# # Dimensions for the tensor x
# N = 3  # Number of batches
# T = 5  # Number of sequences per batch
# D = 3   # Number of features per sequence

# # Create a random 3-dimensional tensor x
# x = torch.randn(N, T, D)
# # print('x start', x)
# # Create some sorted_indices as a 2D tensor
# # Let's assume you want to pick 3 pairs from the tensor
# sorted_indices = torch.tensor([
#     [0, 2],  # From batch 0, pick sequence index 0, 2
#     [1, 4],  # From batch 1, pick sequence index 1, 4
#     [2, 3]   # From batch 2, pick sequence index 2, 3
# ])

# # Extract batch and sequence indices from sorted_indices

# batch_indices = torch.arange(sorted_indices.size(0)).repeat_interleave(sorted_indices.size(1))


# # batch_indices = torch.arange(sorted_indices.size(0))
# sequence_indices = sorted_indices.flatten()

# # Use advanced indexing to retrieve the elements
# selected_elements = x[batch_indices, sequence_indices]

# # Reshape selected_elements back into the desired shape if needed
# selected_elements = selected_elements.view(N, -1, D)

# # Print the original tensor x for reference
# print("Tensor x:")
# print(x)

# # Print the selected elements
# print("\nSelected elements from x using sorted_indices:")
# print(selected_elements)

# batch_indices = torch.arange(N).unsqueeze(1).expand(-1, sorted_indices.size(1))
# print('batch_indices2', batch_indices)
# sequence_indices = sorted_indices

# # Use advanced indexing to retrieve the elements
# selected_elements = x[batch_indices, sequence_indices]
# print('selected_elements2', selected_elements)
# # # Your tensors
# # l2_norms = torch.tensor([1035., 1035., 1035., 1035., 1035., 1035., 1035., 1035., 1035., 1035.,
# #                          1035., 1035., 1035., 1035., 1035., 1035., 1035., 1035., 1035., 1035.],
# #                         device='cuda:0', dtype=torch.float16)
# # noise = torch.tensor([-0.19, 0.08, -0.06, -0.04, -0.09, -0.03, 0.15, 0.05, -0.03, 0.01,
# #                       -0.03, 0.19, -0.16, 0.05, 0.10, -0.09, 0.01, -0.09, 0.08, 0.20],
# #                      device='cuda:0', dtype=torch.float16)
# # print('l2_norms before noise', l2_norms)
# # # Convert both tensors to float32
# # # l2_norms = l2_norms.float()
# # # noise = noise.float()
# # noise = torch.randn(l2_norms.size(), device=l2_norms.device, dtype=l2_norms.dtype) * 0.5
# # # Add noise and print results
# # l2_norms += noise
# # print('l2_norms after noise', l2_norms)


# import argparse

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from LLMPruner.peft import PeftModel


# # Sequence length
# # seq_length = 10
# # bsz = 2
# # # Creating an example attention mask where last 3 elements are masked out (set to 0)
# # attention_mask = torch.ones(bsz, seq_length)
# # attention_mask[:, -3:] = 0
# # print(attention_mask)
# # expanded_mask = attention_mask[:, None, None, :].expand(bsz, 1, seq_length, seq_length)

# # expanded_mask = expanded_mask.transpose(-1, -2)
# # # expanded_mask = attention_mask.unsqueeze(2).repeat(1, 1, seq_length)  # Expand to (bsz, seq_length, seq_length)

# # print(expanded_mask)


# import torch
# import torch.nn.functional as F

# # # Create a tensor with -inf and very large negative values (-65504)
# # values = torch.tensor([-65504., -float('inf')] * 30, dtype=torch.float32)  # Using float16 to match your context

# # values = torch.tensor([-65504.] * 30, dtype=torch.float32)
# # # Apply softmax
# # softmax_output = F.softmax(values, dim=0)

# # print("Softmax Output:", softmax_output)


# import torch


# import torch

# # # Simulating attn_weights with NaN values for demonstration
# # attn_weights = torch.tensor([[0.1, 0.2, 0.3],
# #                              [0.4, float('nan'), 0.6],
# #                              [float('nan'), 0.8, 0.9]])

# # # Find indices where NaN values occur
# # nan_indices = torch.where(torch.isnan(attn_weights))

# # # Printing indices of NaN values
# # print('Row indices of NaN values:', nan_indices[0])
# # print('Column indices of NaN values:', nan_indices[1])
# # print(attn_weights[nan_indices])

# import torch

# # Example tensor representing an 'attention_mask'
# attention_mask = torch.tensor([
#     [1, 1, 1, 0, 0],
#     [1, 1, 0, 0, 0],
#     [1, 1, 1, 1, 1],
#     [0, 1, 1, 1, 0],
#     [1, 0, 0, 0, 0]
# ])

# # Flip each row to make the last '1' be the first '1' from the left
# flipped = torch.flip(attention_mask, dims=[1])
# print("Flipped Tensor:", flipped)
# # Find the index of the first '1' in each flipped row (now the last '1' in the original)
# last_one_indices = torch.argmax(flipped, dim=1)

# # Correct the indices to reflect their positions in the original tensor
# non_padding_tokens = flipped.size(1) - last_one_indices

# print("Last positions of 1 in each row:", correct_indices)


# # Create a 3D tensor of size (3, 4, 5)
# # tensor_3d = torch.randn(3, 4, 5)  # Random values

# # # Create a 2D boolean tensor with the same shape as the first two dimensions of the 3D tensor
# # tensor_2d = torch.rand(3, 4) > 0.5  # Random True/False based on a threshold

# # print("3D Tensor:")
# # print(tensor_3d)
# # print("\n2D Boolean Tensor:")
# # print(tensor_2d)
# # tensor_3d[tensor_2d] = 999
# # print("\nUpdated 3D Tensor:", tensor_3d)
# # Example attention mask tensor (bsz x seq_length)
# # attention_mask = torch.tensor([
# #     [0, 0, 1, 1, 1],
# #     [1, 1, 1, 0, 0],
# #     [0, 1, 1, 1, 1]
# # ])

# # # Find the indices of the first occurrence of 1 in each row
# # first_one_indices = torch.argmax((attention_mask == 1).int(), dim=1).unsqueeze_(1)

# # print(first_one_indices)
# # def load(model_type: str = 'pruneLLM', base_model: str = 'llama2-7b', ckpt: str = '', lora_ckpt: str = ''):
# #     if model_type == 'pruneLLM':
# #         pruned_dict = torch.load(ckpt, map_location='cpu')
# #         tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
# #     elif model_type == 'tune_prune_LLM':
# #         pruned_dict = torch.load(ckpt, map_location='cpu')
# #         tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
# #         model = PeftModel.from_pretrained(
# #             model,
# #             lora_ckpt,
# #             torch_dtype=torch.float16,
# #         )
# #     else:
# #         raise NotImplementedError

# #     if torch.cuda.is_available():
# #         device = "cuda"
# #     else:
# #         device = "cpu"

# #     if device == "cuda":
# #         model.half()
# #         model = model.cuda()

# #     # unwind broken decapoda-research config
# #     model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
# #     model.config.bos_token_id = 1
# #     model.config.eos_token_id = 2
# #     return model, tokenizer

# # base_model = 'pytorch_model.bin'
# # for i in ['results/seed0', 'results/seed1']:
# #     for j in ['llmpruner_prune_tune_block_param1_3_31_0.23_0.2_c4',
# #                 'llmpruner_prune_tune_block_param1_3_31_0.46_0.4_c4',
# #                 'llmpruner_prune_tune_block_param1_3_31_0.68_0.6_c4']:
# #         model, tokenizer = load("pruneLLM", ckpt=i + '/' + j + '/' + base_model)
# #         model.eval()
# #         # ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], 128, device='cuda')
# #         # print(f"pruneLLM from {i + '/' + j}", " PPL after pruning: {}".format(ppl))
# #         # del model

# #         # model, tokenizer = load("tune_prune_LLM", ckpt=i + '/' + j + '/' + base_model, lora_ckpt=i + '/' + j)
# #         # model.eval()
# #         # ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], 128, device='cuda')
# #         # print(f"tune_prune_LLM from {i + '/' + j}", " PPL after pruning: {}".format(ppl))
# #         # del model
# # import numpy as np
# # data_latest = [
# #     [68.4, 77.7, 74.3, 67.8, 41.8, 66.9, 41.3],
# #     [71.1, 78.6, 74.7, 66.3, 42.9, 69.0, 43.1],
# #     [65.9, 76.0, 69.4, 63.8, 36.5, 62.4, 40.3],
# #     [65.3, 77.2, 69.3, 58.4, 38.1, 64.9, 40.3],
# #     [62.8, 71.1, 55.3, 58.6, 31.4, 53.4, 34.8],
# #     [60.8, 71.4, 42.2, 51.8, 29.9, 49.8, 36.4],
# #     [67.7, 75.4, 69.1, 62.7, 38.3, 64.1, 41.0],
# #     [65.0, 76.4, 68.4, 59.3, 39.0, 64.8, 40.6],
# #     [62.5, 70.8, 55.6, 58.5, 33.2, 54.6, 36.2],
# #     [62.0, 72.6, 43.7, 51.0, 29.9, 52.3, 38.5],
# #     [60.6, 70.3, 50.7, 53.8, 30.1, 52.8, 35.9],
# #     [62.0, 70.8, 42.8, 51.0, 29.0, 51.2, 36.9]
# # ]

# # # Calculate the average for each row using numpy for precision
# # averages_latest = np.mean(data_latest, axis=1)
# # print(averages_latest)


# # data_latest = [[81.7, 79.0, 75.1, 79.4, 78.5, 78.7, 78.5],
# #  [77.8, 73.4, 73.5, 76.3, 73.9, 66.0, 67.1],
# #  [70.9, 76.3, 74.8, 69.6, 70.3, 73.1, 70.5],
# #  [56.7, 57.2, 65.3, 59.1, 41.2, 44.0, 50.5],
# #  [72.5, 73.9, 71.4, 72.5, 70.3, 63.2, 66.7],
# #  [67.2, 63.7, 55.0, 57.4, 59.1, 47.2, 44.9],
# #  [38.7, 40.8, 43.2, 42.9, 31.8, 31.7, 39.0],
# #  [35.1, 25.0, 25.9, 32.1, 61.7, 60.7, 56.9],
# #  [57.8, 59.1, 58.9, 47.05, 44.0, 6.0, 57.4],
# #  [54.4, 36.6, 37.1, 49.0, 40.2, 40.2, 39.0],
# #  [39.4, 40.2, 39.8, 34.5, 24.4, 36.9, 35.6],
# #  [29.3, 31.8, 33.9]]

# # # Calculate the average for each row using numpy for precision
# # averages_latest = np.mean(data_latest, axis=1)
# # print(averages_latest)
# import numpy as np
# import re
# data_str = """
# 68.4(0.0)	75.8(0.0)	70.4(0.0)	63.2(0.0)	38.9(0.0)	64.4(0.0)	42.2(0.0)
# 69.8(0.0)	75.7(0.0)	70.7(0.0)	63.7(0.0)	39.2(0.0)	64.9(0.0)	41.4(0.0)
# 69.3(0.0)	76.7(0.0)	70.9(0.0)	63.8(0.0)	40.1(0.0)	65.4(0.0)	40.2(0.0)
# 69.0(0.1)	78.1(0.0)	73.5(0.0)	66.7(0.3)	42.8(0.1)	68.5(0.0)	40.9(0.2)
# 67.3(0.1)	77.8(0.1)	73.7(0.0)	64.8(0.1)	41.5(0.1)	67.4(0.2)	41.3(0.3)
# 68.1(0.1)	77.5(0.1)	73.7(0.0)	66.7(0.3)	42.2(0.1)	68.2(0.1)	42.7(0.4)
# 57.4(0.0)	71.3(0.0)	55.7(0.0)	54.6(0.0)	31.7(0.0)	53.3(0.0)	34.6(0.0)
# 62.1(0.0)	72.1(0.0)	56.9(0.0)	58.3(0.0)	34.3(0.0)	57.9(0.0)	35.4(0.0)
# 63.8(0.0)	72.3(0.0)	57.6(0.0)	56.5(0.0)	33.5(0.0)	57.7(0.0)	36.0(0.0)
# 62.7(0.2)	74.9(0.1)	63.6(0.0)	57.5(0.2)	35.5(0.1)	61.7(0.2)	40.3(0.4)
# 64.3(0.1)	74.5(0.1)	64.2(0.1)	57.9(0.4)	37.6(0.1)	62.9(0.2)	40.7(1.1)
# 64.7(0.1)	74.3(0.1)	64.4(0.1)	58.1(0.3)	37.7(0.3)	62.5(0.1)	41.3(0.2)
# 58.5(0.0)	63.4(0.0)	35.3(0.0)	48.4(0.0)	23.6(0.0)	39.1(0.0)	32.0(0.0)
# 60.7(0.0)	63.6(0.0)	36.1(0.0)	49.7(0.0)	24.9(0.0)	41.8(0.0)	31.8(0.0)
# 61.7(0.0)	66.4(0.0)	36.8(0.0)	50.4(0.0)	25.6(0.0)	42.8(0.0)	32.0(0.0)
# 61.6(0.2)	66.9(0.2)	36.7(0.0)	52.2(0.4)	27.0(0.1)	46.2(0.2)	33.9(0.1)
# 62.0(0.1)	67.6(0.3)	37.2(0.2)	50.6(0.4)	26.9(0.2)	48.2(0.3)	31.5(0.2)
# 62.1(0.0)	68.3(0.3)	37.1(0.0)	51.9(0.3)	27.4(0.2)	48.7(0.1)	33.3(0.1)

# """

# # Extract only the primary numbers not in parentheses
# cleaned_numbers = re.findall(r'(\d+\.\d+)(?=\(\d+\.\d+\))', data_str)
# print('cleaned_numbers', cleaned_numbers)
# # Convert to float for calculations
# cleaned_floats = [float(num) for num in cleaned_numbers]

# # # Organize these into rows of 7 for calculation
# rows = np.array(cleaned_floats).reshape(-1, 7)

# print('rows', rows)
# # Calculate averages for each row
# averages = np.mean(rows, axis=1)

# print('averages', averages)
# data = [
#     [65.3, 77.2, 74.1, 67.1, 41.1, 63.9, 41.8],
#     [68.7, 77.7, 73.7, 64.2, 41.3, 67.7, 41.3],
#     [67.3, 76.6, 73.0, 67.4, 40.6, 63.1, 42.0],
#     [69.9, 77.2, 72.2, 65.3, 41.3, 67.9, 40.0],
#     [69.0, 78.1, 73.5, 66.7, 42.8, 68.5, 40.9],
#     [62.5, 72.5, 63.3, 56.9, 33.4, 54.4, 40.8],
#     [61.4, 74.6, 63.9, 54.9, 36.1, 58.0, 40.1],
#     [63.5, 71.7, 63.3, 59.8, 33.8, 52.5, 40.0],
#     [58.3, 72.5, 62.7, 55.0, 34.8, 56.9, 36.1],
#     [62.7, 74.9, 63.6, 57.5, 35.5, 61.7, 40.3],
#     [61.9, 61.7, 38.4, 51.5, 27.2, 36.0, 30.7],
#     [60.0, 66.9, 40.5, 51.0, 26.7, 41.4, 35.4],
#     [50.9, 63.8, 44.3, 52.1, 26.7, 38.1, 30.8],
#     [55.7, 65.1, 36.4, 50.9, 26.8, 41.0, 30.0],
#     [61.6, 66.9, 36.7, 52.2, 27.0, 46.2, 33.9]
# ]

# # Calculate the average for each row
# averages = [sum(row) / len(row) for row in data]
# print(averages)


# data_new = [
#     [66.0, 75.4, 63.0, 64.8, 33.7, 48.2, 35.0],
#     [66.8, 75.3, 65.3, 65.6, 33.5, 51.9, 36.5],
#     [68.1, 75.1, 62.5, 62.6, 31.8, 49.5, 34.5],
#     [66.8, 75.2, 65.3, 66.1, 33.9, 50.5, 35.7],
#     [67.4, 75.5, 65.7, 64.9, 33.8, 51.6, 36.5],
#     [63.7, 71.8, 53.2, 57.6, 29.6, 43.3, 34.3],
#     [60.7, 74.5, 58.2, 57.8, 33.1, 49.7, 35.4],
#     [62.7, 72.4, 53.3, 58.3, 29.4, 45.2, 34.1],
#     [57.9, 74.4, 58.1, 57.9, 32.2, 49.6, 33.5],
#     [61.1, 74.3, 58.7, 59.3, 33.6, 49.7, 35.3],
#     [42.3, 63.6, 32.5, 49.7, 23.0, 35.5, 29.5],
#     [47.5, 70.6, 39.3, 51.4, 26.8, 45.4, 32.4],
#     [56.9, 63.5, 35.3, 51.1, 25.5, 34.8, 30.3],
#     [47.4, 66.8, 35.8, 51.2, 24.6, 41.7, 30.1],
#     [48.0, 71.4, 42.2, 51.0, 27.1, 45.8, 32.5]
# ]

# # Calculate the average for each row
# averages_new = [sum(row) / len(row) for row in data_new]
# print(averages_new)
# Data setup
# Generate arrays of pruning ratios from 0.1 to 1.0, incrementing by 0.1
# mlp_prune_ratio = np.arange(0.1, 1.1, 0.1)
# attention_prune_ratio = np.arange(0.1, 1.1, 0.1)

# # Weights
# mlp_weight = 0.7
# attention_weight = 0.3

# # Calculate the total pruning ratio using the formula you provided
# # np.outer is used here to apply the multiplication across all combinations of mlp_prune_ratio and attention_prune_ratio
# total_pruning_ratio = np.empty((10, 10))


# for i in range(len(mlp_prune_ratio)):
#     for j in range(len(attention_prune_ratio)):
#         total_pruning_ratio[i, j] = mlp_weight * mlp_prune_ratio[i] + attention_weight * attention_prune_ratio[j]


# # Print the resulting 10x10 matrix
# print("Total Pruning Ratio Matrix:")
# print(total_pruning_ratio)

# # Display the total pruning ratio matrix
# print("Total Pruning Ratio Matrix:")
# print(total_pruning_ratio)

# # Generating perplexity for demonstration
# perplexity = 100 / total_pruning_ratio

# # Creating the heatmap
# plt.figure(figsize=(10, 8))
# ax = sns.heatmap(total_pruning_ratio, annot=perplexity, fmt=".1f", cmap="viridis", xticklabels=np.round(mlp_prune_ratio,1), yticklabels=np.round(attention_prune_ratio,1))
# ax.set_xlabel('MLP Prune Ratio')
# ax.set_ylabel('Attention Prune Ratio')
# ax.set_title('Total Pruning Ratio with Perplexity Annotations')
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Data setup
# mlp_prune_ratio = np.arange(0.1, 1.1, 0.1)
# attention_prune_ratio = np.arange(0.1, 1.1, 0.1)

# # Weights
# mlp_weight = 0.7
# attention_weight = 0.3

# # Calculate the total pruning ratio
# total_pruning_ratio = np.empty((10, 10))
# for i in range(len(mlp_prune_ratio)):
#     for j in range(len(attention_prune_ratio)):
#         total_pruning_ratio[i, j] = mlp_weight * mlp_prune_ratio[i] + attention_weight * attention_prune_ratio[j]

# # Generating perplexity for demonstration
# perplexity = 100 / total_pruning_ratio

# # Creating the heatmap
# plt.figure(figsize=(10, 8))
# ax = sns.heatmap(total_pruning_ratio, annot=perplexity, fmt=".1f", cmap="Reds", 
#                  xticklabels=np.round(mlp_prune_ratio,1), yticklabels=np.round(attention_prune_ratio,1))

# # Adjust the x and y axis to range from 0 to 1
# ax.set_xticklabels(np.round(mlp_prune_ratio,1))
# ax.set_yticklabels(np.round(attention_prune_ratio[::-1],1))  # Reverse the yticklabels for inversion

# ax.set_xlabel('MLP Prune Ratio')
# ax.set_ylabel('Attention Prune Ratio')
# ax.set_title('Total Pruning Ratio with Perplexity Annotations')

# # Invert the y-axis to have 1 at the top
# ax.invert_yaxis()

# plt.show()

# plt.close('all')







# # Example values for self.in_features and async_in_dim_indices
# in_features = 10  # Just an example, set this to your actual in_features
# async_in_dim_indices = torch.tensor([2, 5, 7], dtype=torch.long).to(device='cuda')  # Example indices to exclude

# # Create a tensor of all indices from 0 to in_features - 1
# all_indices = torch.arange(in_features, dtype=torch.long).to(async_in_dim_indices.device)

# # Create a mask where we mark indices that are not in async_in_dim_indices
# mask = ~torch.isin(all_indices, async_in_dim_indices)

# # Apply the mask to filter out the indices
# pruned_channel_indices = all_indices[mask]

# # pruned_channel_indices now contains indices excluding those in async_in_dim_indices
# print(pruned_channel_indices)


# a = torch.tensor(1)
# b = a.reshape(1,-1)
# print(b.shape)
# a = torch.tensor(1, dtype=torch.float16, device='cuda')
# b = a / 1000000000000000
# print('b', b)
# b1 = torch.sqrt(b)
# print('b1', b1)
# c = a * 1000000000000000
# print('c', c)


# # tensor_3d = torch.randn(4, 5, 6)  # Shape (4, 5, 6)
# # norm1 = torch.linalg.vector_norm(tensor_3d, ord=2, dim=(0,1))
# # norm2 = torch.linalg.matrix_norm(tensor_3d, ord='fro', dim=(0,1))
# # print(norm1, norm2)
# # print(norm1 == norm2)
# # norm3 = torch.linalg.vector_norm(tensor_3d, p=2, dim=(0,1))
# # print(norm3)



# tensor_3d = torch.tensor([65500, 65500], dtype=torch.float16, device='cuda')
# print(tensor_3d)
# norm1 = torch.linalg.vector_norm(tensor_3d, ord=2)
# # norm2 = torch.linalg.matrix_norm(tensor_3d, ord='fro', dim=(0,1))
# print(norm1)
# # print(norm1 == norm2)
# norm3 = torch.clamp(torch.linalg.vector_norm(tensor_3d, ord=2), max=65504)
# print(norm3)
#   # Shape (4, 6


# a = torch.tensor(5.5, dtype=torch.float16, device='cuda')
# b = torch.tensor(3.5, dtype=torch.float32, device='cuda')
# c = a + b
# print(c, c.dtype)


# numerator = torch.tensor([0.0])
# denominator = torch.tensor([0.0])

# # Performing division that should result in NaN
# result = numerator / (denominator + 1e-6)
# print(result)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# a = torch.rand(4096, 11008, device=device, dtype=torch.float16)
# b = torch.rand(4096, 11008, device=device, dtype=torch.float16)
# a_large = torch.rand(10000, 10000, device=device, dtype=torch.float16)
# b_large = torch.rand(10000, 10000, device=device, dtype=torch.float16)

# # Create two CUDA streams
# stream1 = torch.cuda.Stream()
# stream2 = torch.cuda.Stream()

# import threading



# def func1():
#     # ----test multiply async
#     with torch.cuda.stream(stream1):
#         for i in range(10000):
#             result1 = torch.multiply(a, a)  # a * a

# def func2():
#     with torch.cuda.stream(stream2):
#         for i in range(10000):
#             result2 = torch.multiply(b, b)  # b * b


# # Create threads
# thread1 = threading.Thread(target=func1)
# thread2 = threading.Thread(target=func2)

# # Start the threads
# thread1.start()
# thread2.start()

# # Wait for both threads to complete
# thread1.join()
# thread2.join()



# torch.cuda.synchronize()


# time.sleep(5)

# def func1():
#     # ----test multiply async
#     with torch.cuda.stream(stream1):
#         for i in range(10000):
#             result1 = torch.multiply(a, a)  # a * a

# def func2():
#     with torch.cuda.stream(stream2):
#         for i in range(10000):
#             result2 = torch.multiply(b, b)  # b * b


# # Create threads
# thread1 = threading.Thread(target=func1)
# thread1.start()
# # thread2 = threading.Thread(target=func2)
# func2()

# # Start the threads

# # thread2.start()

# # Wait for both threads to complete
# thread1.join()
# # thread2.join()



# torch.cuda.synchronize()
# time.sleep(5)

# # ----test multiply async
# with torch.cuda.stream(stream1):
#     for i in range(10000):
#         result1 = torch.multiply(a, a)  # a * a


# with torch.cuda.stream(stream2):
#     for i in range(10000):
#         result2 = torch.multiply(b, b)  # b * b

# torch.cuda.synchronize()
# # # Create threads
# # thread1 = threading.Thread(target=func1)
# # thread2 = threading.Thread(target=func2)

# # # Start the threads
# # thread1.start()
# # thread2.start()

# # # Wait for both threads to complete
# # thread1.join()
# # thread2.join()
# import time
# time.sleep(5)

# # ----test matmul, multiply async
# layer1 = torch.nn.Linear(4096, 11008, device=device)
# layer2 = torch.nn.Linear(11008, 4096, device=device)
# input_tensor = torch.randn(128, 4096, device=device)

# def func1():
#     with torch.cuda.stream(stream1):
#         for i in range(10000):
#             result1 = layer2(layer1(input_tensor))  # layer1(input_tensor)

# def func2():
#     with torch.cuda.stream(stream2):
#         for i in range(10000):
#             result2 = torch.multiply(b, b)  # b * b



# # Create threads
# thread1 = threading.Thread(target=func1)
# thread2 = threading.Thread(target=func2)

# # Start the threads
# thread1.start()
# thread2.start()

# # Wait for both threads to complete
# thread1.join()
# thread2.join()

# torch.cuda.synchronize()

# # ----test matmul, select indices async
# layer1 = torch.nn.Linear(80, 80, device=device)
# layer2 = torch.nn.Linear(80, 80, device=device)
# input_tensor = torch.randn(128, 80, device=device)
# select_indices = torch.arange(500, device=device)

# with torch.cuda.stream(stream1):
#     for i in range(100000):
#         result1 = layer2(layer1(input_tensor))  # layer1(input_tensor)

# with torch.cuda.stream(stream2):
#     for i in range(100000):
#         _ = b[:, select_indices]

# torch.cuda.synchronize()

# # ----test matmul, select indices async (large matrix)
# layer1 = torch.nn.Linear(4000, 4000, device=device)
# layer2 = torch.nn.Linear(4000, 4000, device=device)
# input_tensor = torch.randn(128, 4000, device=device)
# select_indices = torch.arange(5000, device=device)

# with torch.cuda.stream(stream1):
#     for i in range(100000):
#         result1 = layer2(layer1(input_tensor))  # layer1(input_tensor)

# with torch.cuda.stream(stream2):
#     for i in range(100000):
#         _ = b_large[:, select_indices]

# torch.cuda.synchronize()
exit()















# # -*- coding: utf-8 -*-
# # import torch
# # import numpy as np

# # a = torch.tensor([])

# # b = [a]
# import collections
# # print(len(b))


# # mask = torch.ones(10, dtype=torch.bool)
# # print('pre_mask', mask)
# # # Mark the indices to be pruned as False
# # mask[None] = False
# # print('mask', mask)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch

# import torch


# import torch

# import torch
# import torch.nn as nn
# # from transformers import AutoTokenizer



# import torch
# import time


# import torch
# import torch.nn as nn
# torch.backends.cudnn.benchmark = False


# # import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # # Assume we are using float16 for the computation
# # datamin = torch.finfo(torch.float16).min
# # dataminpositive = torch.finfo(torch.float16).tiny
# # print("Minimum value for float16:", datamin)
# # print("Minimum positive value for float16:", dataminpositive)

# # x = torch.tensor([1e-6], dtype=torch.float16, device=device) / 10

# # # Check if the result is NaN
# # print("Result of division:", x)

# # # Use clamp on the result with a min range of 1e-5
# # clamped_x = torch.clamp(x, min=1e-5)

# # print("Result after clamping:", clamped_x)


# # x = torch.tensor([1e-10], dtype=torch.float16, device=device) / 10

# # # Check if the result is NaN
# # print("Result of division:", x)

# # # Use clamp on the result with a min range of 1e-5
# # clamped_x = torch.clamp(x, min=1e-5)

# # print("Result after clamping:", clamped_x)


# # x = torch.tensor(1e-10, dtype=torch.float16, device=device)
# # x = x * x

# # # Check if the result is NaN
# # print("Result of division:", x)

# # # Use clamp on the result with a min range of 1e-5
# # clamped_x = torch.clamp(x, min=1e-5)

# # print("Result after clamping:", clamped_x)









# # exit()
# # import torch
# # lowest_priority, highest_priority = torch.cuda.get_device_properties(0).priority_range
# # print(f"Priority range: {lowest_priority} (highest priority) to {highest_priority} (lowest priority)")
# # start_event = torch.cuda.Event(enable_timing=True)
# # end_event = torch.cuda.Event(interprocess=True) # what I want to share between the streams by ipc_handle
# # # end_event_ipc_handle = end_event.ipc_handle()
# # pin1_event = torch.cuda.Event(enable_timing=True)
# # pin2_event = torch.cuda.Event(enable_timing=True)

# # with torch.cuda.stream(torch.cuda.Stream()):
# #     start_event.record()
    
# #     # Run some things here
    
# #     pin1_event.record()
# #     end_event.record()

# # with torch.cuda.stream(torch.cuda.Stream()):
# #     # end_event = torch.cuda.Event.from_ipc_handle(torch.cuda.current_device(), end_event_ipc_handle)
# #     end_event.wait() # wait asynchronously

# #     # Run some things here

# #     pin2_event.record()

# # torch.cuda.synchronize()

# # elapsed_time_ms = start_event.elapsed_time(pin1_event)
# # print(f"Elapsed time: {elapsed_time_ms} ms")

# # elapsed_time_ms = pin1_event.elapsed_time(pin2_event)
# # print(f"Elapsed time: {elapsed_time_ms} ms")


# # default_stream = torch.cuda.default_stream()
# # print(default_stream)

# # another_steam = torch.cuda.Stream()
# # print(another_steam)


# class CustomModel(nn.Module):
#     def __init__(self, in_shape=4096, out_shape=11008, device='cuda:0'):
#         super(CustomModel, self).__init__()
#         # Initialize the large matrix as a parameter
#         # self.large_matrix = nn.Parameter(torch.randn(rows, cols, device=device))

#         self.linear1 = torch.nn.Linear(in_shape, out_shape, device=device)
#         self.linear2 = torch.nn.Linear(out_shape, in_shape, device=device)
#         # Optionally, you can set `requires_grad=False` if you don't want to update this matrix during training
        
#     def forward(self, indices):
#         """
#         Extracts columns from the large matrix based on the provided indices.
#         Indices should be a tensor of column indices to extract.
#         """
#         # Ensure indices are on the same device as the matrix
#         indices = indices.to(self.large_matrix.device)
#         # Extract columns based on indices
#         extracted_columns = self.large_matrix[:, indices]
#         return extracted_columns

# # # Example usage

# model = CustomModel(device=device)
# model.to(device)  # Move model to the appropriate device
# # x = torch.randn(10, 128, 4096).to(device)
# # x_sample = torch.randn(1, 128, 4096).to(device)

# # # Function to measure extraction time
# # def matrix_multiplication(x):
# #     start = time.time()
# #     # Ensure the operation is performed on the GPU
# #     _ = model.linear1.weight.to(device)
# #     _ = F.linear(x, model.linear1.weight)
# #     # torch.cuda.synchronize()  # Wait for GPU operations to complete
# #     return time.time() - start



# # def measure_matrix_elementwise(calib):
# #     torch.cuda.synchronize()
# #     weight = model.linear2.weight
# #     torch.cuda.nvtx.range_push("iteration{}".format(0))
# #     squared_weight = torch.pow(weight, 2)
# #     squared_weight = weight.pow(2)
# #     # squared_weight = torch.mul(weight, weight)
# #     torch.cuda.synchronize()
# #     torch.cuda.nvtx.range_pop()

# #     # torch.cuda.nvtx.range_push("iteration{}".format(10))
# #     # squared_weight = torch.abs(weight)
# #     # torch.cuda.nvtx.range_pop()

# #     torch.cuda.nvtx.range_push("iteration{}".format(1))
# #     print('squared_weight', squared_weight.shape)
# #     calib_reshaped = calib.reshape((1 ,-1))
# #     torch.cuda.synchronize()
# #     torch.cuda.nvtx.range_pop()
# #     torch.cuda.nvtx.range_push("iteration{}".format(2))
# #     mult_result = torch.mul(calib_reshaped, squared_weight)
# #     # mult_result = torch.multiply(calib_reshaped, squared_weight)    
# #     torch.cuda.synchronize()
# #     torch.cuda.nvtx.range_pop()
# #     torch.cuda.nvtx.range_push("iteration{}".format(3))

# #     probe_out_dim_metric = torch.sqrt(torch.clamp(torch.sum(mult_result, dim=0), 0, 10))
# #     torch.cuda.synchronize()
# #     # torch.cuda.nvtx.range_pop()
# #     # torch.cuda.nvtx.range_push("iteration{}".format(4))

# #     # # Use torch.sum instead of .sum(), then apply torch.clamp and finally compute the square root
    
# #     # torch.cuda.synchronize()
# #     torch.cuda.nvtx.range_pop()
# #     # torch.cuda.nvtx.range_push("iteration{}".format(4))
# #     torch.cuda.synchronize()  # Wait for GPU operations to complete
# #     return 

# # # Extraction sizes and their corresponding times
# extraction_sizes = [640, 1280, 2560, 4096]
# # sorted_times = []
# # unsorted_times = []
# # # stream1 = torch.cuda.Stream()
# # # stream2 = torch.cuda.Stream()

# #     # start_time = time.time()
# #     # with torch.cuda.stream(stream1):
# #     #     time_sorted = measure_extraction_time(sorted_indices)
 
# #     # with torch.cuda.stream(stream2):
# #     #     time_unsorted = measure_extraction_time(unsorted_indices)
    
# #     # # 
# #     # # torch.cuda.synchronize(stream1)
# #     # # torch.cuda.synchronize(stream2)
# #     # end_time_1 = time.time() - start_time

# # # for size in extraction_sizes:
# # #     # Sorted indices


# # #     sorted_indices = torch.arange(size, device=device)
    
# # #     # Unsorted indices: Shuffle the sorted indices
# # #     unsorted_indices = sorted_indices[torch.randperm(size, device=device)]
# # #     # print('sorted_indices', sorted_indices)
# # #     # print('unsorted_indices', unsorted_indices)
# # #     # Measure time for sorted indices
# # #     time_sorted = matrix_multiplication(x)

# # #     start_time = time.time()
# # #     time_sorted = matrix_multiplication(x)
# # #     time_sorted = matrix_multiplication(x)
# # #     time_sorted = matrix_multiplication(x)
# # #     measure_row_extraction_time(unsorted_indices)
# # #     measure_row_extraction_time(unsorted_indices)
# # #     # sorted_times.append(time_sorted)
    
# # #     # Measure time for unsorted indices
# # #     # time_unsorted = matrix_multiplication(unsorted_indices)
# # #     # time_sorted = matrix_multiplication(sorted_indices)
# # #     # time_sorted = matrix_multiplication(sorted_indices)
# # #     # unsorted_times.append(time_unsorted)
# # #     torch.cuda.synchronize()
# # #     end_time = time.time() - start_time

# # #     # stream2 = torch.cuda.Stream()
# # #     start_time = time.time()
# # #     # with torch.cuda.stream(stream1):
# # #     with torch.cuda.stream(stream1):
# # #         time_sorted = matrix_multiplication(x_sample)
# # #         time_sorted = matrix_multiplication(x_sample)
# # #         time_sorted = matrix_multiplication(x_sample)
# # #         measure_row_extraction_time(unsorted_indices)
# # #         measure_row_extraction_time(unsorted_indices)
    
# # #     time_unsorted = matrix_multiplication(x)
# # #     time_sorted = matrix_multiplication(x)
# # #     time_sorted = matrix_multiplication(x)
# # #     torch.cuda.synchronize(stream1)
# # #     torch.cuda.synchronize()
# # #     end_time_2 = time.time() - start_time

# # #     # total time with stream: {end_time_1:.6f} seconds,
# # #     print(f"Extraction size: {size}, total time: {end_time:.6f} seconds,  total time with one stream: {end_time_2:.6f} seconds")
# #     # print(f"Extraction size: {size}, Time (sorted): {time_sorted:.6f} seconds, Time (unsorted): {time_unsorted:.6f} seconds")

# # extraction_sizes = [640, 1280, 2560, 4096, 8808]
# # sorted_times = []
# # unsorted_times = []

# # for size in extraction_sizes:
# #     # Sorted indices
# #     sorted_indices = torch.arange(size, device=device)
    
# #     # Unsorted indices: Shuffle the sorted indices
# #     unsorted_indices = sorted_indices[torch.randperm(size, device=device)]
    
# #     # Measure time for sorted indices
# #     time_sorted = measure_row_extraction_time(sorted_indices)
# #     sorted_times.append(time_sorted)
    
# #     # # Measure time for unsorted indices
# #     # time_unsorted = measure_row_extraction_time(unsorted_indices)
# #     # unsorted_times.append(time_unsorted)

# #     print(f"Initialization Extraction size: {size}, Time (sorted): {time_sorted:.6f} seconds")

# # for size in extraction_sizes:
# #     # Sorted indices
# #     sorted_indices = torch.arange(size, device=device)
    
# #     # Unsorted indices: Shuffle the sorted indices
# #     unsorted_indices = sorted_indices[torch.randperm(size, device=device)]
    
# #     # Measure time for sorted indices
# #     # time_sorted = measure_row_extraction_time(sorted_indices)
# #     # sorted_times.append(time_sorted)
    
# #     # Measure time for unsorted indices
# #     time_unsorted = measure_row_extraction_time(unsorted_indices)
# #     unsorted_times.append(time_unsorted)

# #     print(f"Extraction size: {size}, Time (unsorted): {time_unsorted:.6f} seconds")



# from torch.utils.data import Dataset, DataLoader
# def measure_row_extraction_time(indices):
#     start = time.time()
#     # Ensure the operation is performed on the GPU
#     _ = model.linear1.weight[indices, :].to(device)
#     return time.time() - start


# def measure_row_extraction_time_indexselect(indices):
#     torch.cuda.synchronize()
#     start = time.time()
#     # Ensure the operation is performed on the GPU
#     # Using torch.index_select to select rows. Change dim=0 to dim=1 if you want to select columns.
#     print(model.linear1.weight.shape, model.linear1.weight.dim(), flush=True)
#     _ = torch.index_select(model.linear1.weight, dim=0, index=torch.tensor(indices).to(device)).to(device)
#     torch.cuda.synchronize()  # Wait for GPU operations to complete
#     end = time.time()
#     return end - start

# default_stream = torch.cuda.default_stream()
# stream1 = torch.cuda.Stream()
# class LargeDataset(Dataset):
#     def __init__(self, total_batches, batch_size, data_shape):
#         # Total dataset size is total_batches * batch_size
#         self.data_len = total_batches * batch_size
#         self.data_shape = data_shape

#     def __len__(self):
#         return self.data_len

#     def __getitem__(self, idx):
#         # Return data of the specified shape (here, it's just random data)
#         return torch.randn(self.data_shape)

# total_batches = 30
# batch_size = 5
# data_shape = (80, 80)

# # Create the dataset
# dataset = LargeDataset(total_batches, batch_size, data_shape)

# # Create the DataLoader
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)



# # 4096 * 11008
# gate_proj = torch.nn.Linear(80, 11008, device=device)
# up_proj = torch.nn.Linear(80, 11008, device=device)
# silu = torch.nn.SiLU(device)
# down_proj = torch.nn.Linear(11008, 80, device=device)

# # matrix multiplicatoin
# with torch.cuda.stream(stream1):
#     for i in range(10):
#         for j, batch in enumerate(dataloader):
#             batch = batch.to(device)
#             res = down_proj(silu(gate_proj(batch)) * up_proj(batch))

# # matrix extractoin
# with torch.cuda.stream(default_stream):
#     for size in extraction_sizes:
#         # Sorted indices
#         for i in range(40):
#             sorted_indices = torch.arange(size, device=device)
#             # Unsorted indices: Shuffle the sorted indices
#             unsorted_indices = sorted_indices[torch.randperm(size, device=device)]
#             time_sorted = measure_row_extraction_time(sorted_indices)


# a[:, list]

import torch

# # Assuming CUDA is available
# assert torch.cuda.is_available()

# Custom matrix multiplication function
# def custom_matmul(a, b):
#     # Simple implementation of matrix multiplication
#     # Note: This is not optimized and is for demonstration purposes only
#     # assert a.shape[1] == b.shape[0]
#     # result = torch.zeros(a.shape[0], b.shape[1], device='cuda')
#     # for i in range(a.shape[0]):
#     #     for j in range(b.shape[1]):
#     #         for k in range(a.shape[1]):
    
#     # torch.select
#     # _ = a[:, unsorted_indices]
#     _ = torch.multiply(a, b)
#     return
#     # return torch.matmul(a, b)
# sorted_indices = torch.arange(500, device='cuda')
    
# # # #     # Unsorted indices: Shuffle the sorted indices
# unsorted_indices = sorted_indices[torch.randperm(500, device='cuda')]
# # Create two tensors on GPU
# a = torch.rand(1000, 1000, device='cuda')  # Reduced size for simplicity
# b = torch.rand(1000, 1000, device='cuda')
# c = torch.rand(1000, 1000, device='cuda')
# d = torch.rand(1000, 1000, device='cuda')

# # Create two CUDA streams
# stream1 = torch.cuda.Stream()
# stream2 = torch.cuda.Stream()

# # Perform first matrix multiplication in stream1
# with torch.cuda.stream(stream1):
#     for i in range(60000):
#         result1 = custom_matmul(a, b)

# # Perform second matrix multiplication in stream2
# with torch.cuda.stream(stream2):
#     for i in range(60000):
#         result2 = custom_matmul(c, d)

# # Synchronize streams if necessary before accessing results
# torch.cuda.synchronize()

iterations = 1000000
# import torch

# # Ensure CUDA is available
# assert torch.cuda.is_available()

# # Create two large tensors
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# a = torch.rand(20000, 20000, device=device, dtype=torch.float16)
# b = torch.rand(20000, 20000, device=device, dtype=torch.float16)
# # print(a.dtype)
# # Create two CUDA streams
# stream1 = torch.cuda.Stream()
# stream2 = torch.cuda.Stream()

# # Operation 1 in stream1
# with torch.cuda.stream(stream1):
#     for i in range(iterations):
#         result1 = torch.multiply(a, a)  # a * a
       
# # Operation 2 in stream2
# with torch.cuda.stream(stream2):
#     for i in range(iterations):
#         result2 = torch.multiply(b, b)  # b * b

# # Wait for all streams to complete
# torch.cuda.synchronize()
# print('result1', result1.device)



import torch


from numba import cuda
import numpy as np

# Define a CUDA kernel using Numba
@cuda.jit
def multiply_kernel(x, y, out):
    # Calculate the index of the current thread in the grid
    tx = cuda.threadIdx.x  # Thread index in the block
    ty = cuda.blockIdx.x   # Block index in the grid
    bw = cuda.blockDim.x   # Number of threads per block

    # Calculate the flat index
    index = ty * bw + tx

    # Check if the index is within the bounds of the arrays
    if index < x.size:
        out[index] = x[index] * y[index]

# Example usage
def multiply_arrays(x, y):
    # Ensure the arrays are of the same size and type
    assert x.size == y.size
    assert x.dtype == y.dtype
    
    # Output array
    out = np.zeros_like(x)

    # Number of threads per block
    threads_per_block = 1024
    # Number of blocks in the grid
    blocks = (x.size + (threads_per_block - 1)) // threads_per_block

    # Launch the CUDA kernel
    multiply_kernel[blocks, threads_per_block](x, y, out)

    return out



@cuda.jit
def multiply_2d_kernel(x, y, out):
    # Calculate the row and column index for each thread
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Check if the row and column are within the bounds of the matrices
    if row < x.shape[0] and col < x.shape[1]:
        out[row, col] = x[row, col] * y[row, col]

# Example usage for 2D matrices
def multiply_matrices(x, y):
    # Ensure the arrays are of the same size and type
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    
    # Output array
    out = np.zeros_like(x)

    # Define the number of threads in a block
    threads_per_block = (16, 16)  # 16x16 is a common block size
    # Calculate the number of blocks in the grid
    blocks_in_grid_x = (x.shape[1] + (threads_per_block[1] - 1)) // threads_per_block[1]
    blocks_in_grid_y = (x.shape[0] + (threads_per_block[0] - 1)) // threads_per_block[0]

    # Launch the CUDA kernel
    multiply_2d_kernel[(blocks_in_grid_x, blocks_in_grid_y), threads_per_block](x, y, out)

    return out
# Create example data
# x = np.random.rand(10000).astype(np.float32)
a = torch.rand(10000, 10000, device='cuda')
b = torch.rand(10000, 10000, device='cuda')
c = torch.rand(1000, 1000, device='cuda')
# y = np.random.rand(10000).astype(np.float32)

stream1 = torch.cuda.Stream()
with torch.cuda.stream(stream1):
    for i in range(1000000):
        result1 = torch.multiply(c, c)  # a * a

# Multiply arrays on the GPU
for i in range(10):
    result = multiply_matrices(a.cpu().numpy(), b.cpu().numpy())



# # Ensure CUDA is available
# assert torch.cuda.is_available()

# # Create two large tensors
# a = torch.rand(1000, 1000, device='cuda')
# b = torch.rand(1000, 1000, device='cuda')

# # Create two CUDA streams
# stream1 = torch.cuda.Stream()
# stream2 = torch.cuda.Stream()

# # Operation 1 in stream1
# with torch.cuda.stream(stream1):
#     for i in range(iterations):
#         result1 = torch.multiply(a, a)  # a * a

# # Operation 2 in stream2
# with torch.cuda.stream(stream2):
#     for i in range(iterations):
#         result2 = torch.multiply(b, b)  # b * b

# # Wait for all streams to complete
# torch.cuda.synchronize()
# print(result1, result2)

# import torch

# # Assuming CUDA is available
# assert torch.cuda.is_available()

# # Create two tensors on GPU
# a = torch.rand(1000, 1000, device='cuda')
# b = torch.rand(1000, 1000, device='cuda')
# c = torch.rand(1000, 1000, device='cuda')
# d = torch.rand(1000, 1000, device='cuda')

# # Create two CUDA streams
# stream1 = torch.cuda.Stream()
# stream2 = torch.cuda.Stream()

# # Perform first matrix multiplication in stream1
# with torch.cuda.stream(stream1):
#     for i in range(50):
#         result1 = torch.matmul(a, b)

# # Perform second matrix multiplication in stream2
# with torch.cuda.stream(stream2):
#     for i in range(50):
#         result2 = torch.matmul(c, d)
#     # result2 = torch.matmul(c, d)

# # Synchronize streams if necessary before accessing results
# # This ensures that the matrix multiplications are complete
# torch.cuda.synchronize()

# print(result1, result2)



























exit()
# torch.cuda.nvtx.range_push("iteration{}".format(100))
# for size in extraction_sizes:
#     # Sorted indices
#     sorted_indices = torch.arange(size, device=device)
    
#     # Unsorted indices: Shuffle the sorted indices
#     unsorted_indices = sorted_indices[torch.randperm(size, device=device)]
#     torch.cuda.nvtx.range_push("iteration{}".format(2))
#     # Measure time for sorted indices
#     if device.type == 'cuda':
#         print(f"Memory Allocated: {torch.cuda.memory_allocated(device)} bytes")
#         print(f"Memory Cached (Reserved): {torch.cuda.memory_reserved(device)} bytes")
#         print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(device)} bytes")
#         print(f"Max Memory Cached (Reserved): {torch.cuda.max_memory_reserved(device)} bytes")
#     time_sorted = measure_row_extraction_time_indexselect(sorted_indices)

    # torch.cuda.nvtx.range_pop()
    # sorted_times.append(time_sorted)
    
    # # Measure time for unsorted indices
    # time_unsorted = measure_row_extraction_time(unsorted_indices)
    # unsorted_times.append(time_unsorted)

    # print(f"torch select Extraction size: {size}, Time (sorted): {time_sorted:.6f} seconds")



# tensor = torch.randn(40, 1024, 4096, dtype=torch.float16, device=device)
# torch.cuda.nvtx.range_push("iteration{}".format(3))
# tensor = tensor.to(torch.float32)
# torch.cuda.nvtx.range_pop()
# torch.cuda.nvtx.range_push("iteration{}".format(4))
# tensor = torch.clamp(tensor, 0, 10)
# torch.cuda.nvtx.range_pop()
# torch.cuda.nvtx.range_push("iteration{}".format(5))
# tensor = torch.linalg.vector_norm(tensor, p=2, dim=(0,1)) ** 2
# torch.cuda.nvtx.range_pop()

# torch.cuda.nvtx.range_pop()
# torch.cuda.nvtx.range_push("iteration{}".format(200))
# for size in extraction_sizes:
#     # Sorted indices
#     sorted_indices = torch.arange(size, device=device)
    
#     # Unsorted indices: Shuffle the sorted indices
#     unsorted_indices = sorted_indices[torch.randperm(size, device=device)]
    
#     # Measure time for sorted indices
#     time_sorted = measure_row_extraction_time_indexselect(unsorted_indices)
#     sorted_times.append(time_sorted)
    
#     # # Measure time for unsorted indices
#     # time_unsorted = measure_row_extraction_time(unsorted_indices)
#     # unsorted_times.append(time_unsorted)

#     print(f"torch select Extraction size: {size}, Time (unsorted_indices): {time_sorted:.6f} seconds")
# torch.cuda.nvtx.range_pop()

# random_floats_tensor = torch.randn(11008, device=device)
# measure_matrix_elementwise(random_floats_tensor)


# scalar_inp = torch.randn(128, 11008, device=device)
# index = torch.arange(5000, device=device)
# update = torch.randn(128, 5000, device=device)
# scalar_inp[:, index] += update * 0.99




def measure_matrix_elementwise(calib):
    torch.cuda.synchronize()
    weight = model.linear2.weight
    torch.cuda.nvtx.range_push("iteration{}".format(0))
    squared_weight = torch.pow(weight, 2)
    squared_weight = weight.pow(2)
    # squared_weight = torch.mul(weight, weight)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # torch.cuda.nvtx.range_push("iteration{}".format(10))
    # squared_weight = torch.abs(weight)
    # torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("iteration{}".format(1))
    print('squared_weight', squared_weight.shape)
    calib_reshaped = calib.reshape((1 ,-1))
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("iteration{}".format(2))
    mult_result = torch.mul(calib_reshaped, squared_weight)
    # mult_result = torch.multiply(calib_reshaped, squared_weight)    
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("iteration{}".format(3))

    probe_out_dim_metric = torch.sqrt(torch.clamp(torch.sum(mult_result, dim=0), 0, 10))
    torch.cuda.synchronize()
    # torch.cuda.nvtx.range_pop()
    # torch.cuda.nvtx.range_push("iteration{}".format(4))

    # # Use torch.sum instead of .sum(), then apply torch.clamp and finally compute the square root
    
    # torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    # torch.cuda.nvtx.range_push("iteration{}".format(4))
    torch.cuda.synchronize()  # Wait for GPU operations to complete
    return 



# Iterating over the DataLoader


# for i, batch in enumerate(dataloader):
#     batch = batch.to(device)
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))

# # Parameters
# total_batches = 30
# batch_size = 100
# data_shape = (128, 4096)

# # Create the dataset
# dataset = LargeDataset(total_batches, batch_size, data_shape)

# # Create the DataLoader
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# gate_proj = torch.nn.Linear(4096, 11008, device=device)
# up_proj = torch.nn.Linear(4096, 11008, device=device)
# silu = torch.nn.SiLU(device)
# down_proj = torch.nn.Linear(11008, 4096, device=device)
# # Iterating over the DataLoader
# for i, batch in enumerate(dataloader):
#     batch = batch.to(device)
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
# torch.cuda.empty_cache()



# torch.cuda.empty_cache()

# def nml_process(x, probe_num, probe_size):
#     # avoid nan proportion
#     abs_x = torch.clamp(torch.abs(x), min=1e-6)
#     sum_across_bsz = abs_x.view(probe_num, probe_size, x.size(-2), x.size(-1)).sum(dim=1, keepdim=True)
#     proportion = abs_x.view(probe_num, probe_size, x.size(-2), x.size(-1)) / sum_across_bsz
#     comp_across_bsz = (x.view(probe_num, probe_size, x.size(-2), x.size(-1)) * proportion).sum(dim=1)
#     return comp_across_bsz

# stream1 = torch.cuda.Stream()
# for i, batch in enumerate(dataloader):
#     batch = batch.to(device)
#     torch.cuda.nvtx.range_push("iteration{}".format(3))
#     with torch.cuda.stream(stream1):
#         temp = nml_process(batch, 1, 50)
#     # temp = nml_process(batch, 1, 50)
#     torch.cuda.nvtx.range_pop()

#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))




# # List of all layers
# layers = [gate_proj, up_proj, down_proj]  # SiLU has no parameters

# # Set requires_grad to False for all parameters in these layers
# for layer in layers:
#     for param in layer.parameters():
#         param.requires_grad = False
# stream1 = torch.cuda.Stream()
# for i, batch in enumerate(dataloader):
#     batch = batch.to(device)
#     torch.cuda.nvtx.range_push("iteration{}".format(3))
#     with torch.cuda.stream(stream1):
#         temp = nml_process(batch, 1, 50)
#     # temp = nml_process(batch, 1, 50)
#     torch.cuda.nvtx.range_pop()

#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))


# total_batches = 30
# batch_size = 10
# data_shape = (128, 4096)

# # Create the dataset
# dataset = LargeDataset(total_batches, batch_size, data_shape)

# # Create the DataLoader
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# for i, batch in enumerate(dataloader):
#     batch = batch.to(device)
#     torch.cuda.nvtx.range_push("iteration{}".format(3))
#     with torch.cuda.stream(stream1):
#         temp = nml_process(batch, 1, 10)
#     # temp = nml_process(batch, 1, 50)
#     torch.cuda.nvtx.range_pop()

#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))

    # torch.cuda.nvtx.range_push("iteration{}".format(4))
    # batch = torch.mean(batch)
    # torch.cuda.nvtx.range_pop()

    
    # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
    # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
    # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
    # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
    # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))

# for i, batch in enumerate(dataloader):
#     batch = batch.to(device)
#     batch = torch.mean()
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
    # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
    # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
    # res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
torch.cuda.empty_cache()



# batch_size = 10
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# gate_proj = torch.nn.Linear(4096, 11008, device=device)
# up_proj = torch.nn.Linear(4096, 11008, device=device)
# silu = torch.nn.SiLU(device)
# down_proj = torch.nn.Linear(11008, 4096, device=device)
# # Iterating over the DataLoader
# for i, batch in enumerate(dataloader):
#     batch = batch.to(device)
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))

# torch.cuda.empty_cache()
# batch_size = 400
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# gate_proj = torch.nn.Linear(4096, 11008, device=device)
# up_proj = torch.nn.Linear(4096, 11008, device=device)
# silu = torch.nn.SiLU(device)
# down_proj = torch.nn.Linear(11008, 4096, device=device)
# # Iterating over the DataLoader
# for i, batch in enumerate(dataloader):
#     batch = batch.to(device)
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
#     res = down_proj(silu(gate_proj(batch)) * up_proj(batch))
# input_tensor_1 = torch.randn(1, 4096, device=device)

# out1 = F.linear(input_tensor_1, weight_matrixs.weight)

# out1 = F.linear(input_tensor_1, weight_matrixs.weight)

# out1 = F.linear(input_tensor_1, weight_matrixs.weight)

# out1 = F.linear(input_tensor_1, weight_matrixs.weight)

# out1 = F.linear(input_tensor_1, weight_matrixs.weight)
# result = torch.einsum('ij,ij->ij', weight_matrixs.weight, weight_matrixs.weight)

# result = (weight_matrixs.weight * weight_matrixs.weight)


# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ]
# ) as p:
#     result = (weight_matrixs.weight * weight_matrixs.weight)
#     result = (weight_matrixs.weight * weight_matrixs.weight)
#     result = (weight_matrixs.weight * weight_matrixs.weight)
# print(p.key_averages().table(
#     sort_by="self_cuda_time_total", row_limit=-1))
# time = time.start()
# print()
# torch.cuda.synchronize()
# for i in range(10):
#     start = time.time()
#     out1 = F.linear(input_tensor_1, weight_matrixs.weight)
#     end_time = time.time() - start
#     print(f"Time taken: {end_time:.6f} seconds")

# for i in range(10):
#     start = time.time()
#     out1 = input_tensor_1 * weight_matrixs.weight
#     end_time = time.time() - start
#     print(f"Time taken: {end_time:.6f} seconds")

# start = time.time()
# out1 = F.linear(input_tensor_1, weight_matrixs.weight)
# end_time = time.time() - start
# print(f"Time taken: {end_time:.6f} seconds")

# start = time.time()
# out1 = F.linear(input_tensor_1, weight_matrixs.weight)
# end_time = time.time() - start
# print(f"Time taken: {end_time:.6f} seconds")

# out2 = input_tensor_1 * weight_matrixs.weight

# input_tensor_1 = torch.randn(1, 4096, device=device)
# weight_matrixs = torch.nn.Linear(4096, 4096, device=device)
# out1 = F.linear(input_tensor_1, weight_matrixs.weight)
# out2 = input_tensor_1 * weight_matrixs.weight
# # weight = model.linear2.weight
# random_floats_tensor = torch.randn(1, 128, 11008, device=device)
# norm_l2 = torch.pow(torch.linalg.norm(random_floats_tensor, ord=2, dim=(0, 1)), 2)
# mult_result = torch.mul(norm_l2.reshape(1, -1), weight)
# probe_out_dim_metric = torch.sqrt(torch.clamp(torch.sum(mult_result, dim=0), 0, 10))


# norm_l2 = torch.pow(torch.linalg.norm(random_floats_tensor, ord=2, dim=(0, 1)), 2)
# interleaved_norm_l2 = norm_l2.repeat_interleave(128)

# # Reshape the interleaved tensor to (128, -1)
# reshaped_norm_l2 = interleaved_norm_l2.reshape(128, -1)
# matmul_res = F.linear(random_floats_tensor, model.linear2.weight)



# input = torch.randn(2, 3, 4)
# weight = torch.randn(4, 5)

# norm_bsz_seq_input =  torch.linalg.vector_norm(input, ord=2, dim=(0,1)).un

# import matplotlib.pyplot as plt
# import numpy as np

# # Define the function f(x)
# def f(x):
#     return x / (631.25 + x) * 0.25 + 0.001 * x**2 / (3360 + 0.001 * x**2) * 0.6

# # Generate x values from 0 to 1000
# x = np.linspace(0, 7000, 7000)

# # Calculate y values using the defined function
# y = f(x)

# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(x, y, label='f(x) = x/(631.25+x) * 0.25 + 0.001 * x^2 / (3360+0.001*x^2) * 0.6')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('Plot of the function f(x)')
# plt.grid(True)
# plt.legend()

# # Show the plot
# plt.show()


# # for i in range(173):
# #     cur_number = i * 64
# #     ratio = cur_number / 11008
# #     print(ratio)


# # a = np.std([5], axis=0).item()


# # # Load a pre-trained tokenizer
# # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# # # List of strings
# # texts = ["Hello, world!", "Transformers are great for NLP tasks."]

# # # Tokenize the list of strings
# # tokenized_outputs = tokenizer(texts, padding=True, return_tensors="pt")

# # print(tokenized_outputs)
# # a = np.array([-5, -4])
# # c = np.argmax(a)
# # # Define CrossEntropyLoss with no reduction
# # loss_fn = nn.CrossEntropyLoss(reduction='none')

# # # Example input logits (batch size of 3, 4 classes)
# # logits = torch.tensor([[2.0, 0.5, 1.0, 0.2],
# #                        [0.5, 2.0, 1.0, 0.2],
# #                        [0.2, 0.5, 2.0, 1.0]])

# # # Corresponding labels (batch size of 3)
# # labels = torch.tensor([0, 1, 2])

# # # Calculate loss
# # loss = loss_fn(logits, labels)

# # print(loss)

# # modified_logs = F.log_softmax(logits, dim=-1)
# # print(modified_logs)

# # a = []
# # # a.extend(6)
# # a.extend([6])
# # a.extend([7])
# # print(a)
# # # Sample input
# # bsz = 4  # Example batch size
# # h = torch.randn(bsz, 10)  # Example tensor with shape [batch_size, features]

# # # Simulated layer weights
# # layer_info = {'weight': torch.randn(10)}

# # # First piece
# # sum_squared_norms = torch.sum(torch.linalg.vector_norm(h, p=2, dim=1) ** 2, dim=0)
# # average_squared_norm = sum_squared_norms / torch.tensor(bsz, device=h.device, dtype=torch.float)
# # norm_across_other_dims_first = (torch.sqrt(average_squared_norm.unsqueeze(0).reshape((1,-1))) * torch.abs(layer_info['weight'])).sum(dim=0)

# # # Second piece
# # scaler_inp = torch.zeros_like(sum_squared_norms)
# # nsamples = 0
# # for i in range(bsz):
# #     scaler_inp *= nsamples / (nsamples + 1)
# #     temp = torch.linalg.vector_norm(h[i], p=2, dim=0) ** 2
# #     scaler_inp +=  temp / (nsamples + 1)
# #     nsamples += 1
# # norm_across_other_dims_second = (torch.sqrt(scaler_inp.unsqueeze(0).reshape((1,-1))) * torch.abs(layer_info['weight'])).sum(dim=0)

# # # Compare the results
# # are_equivalent = torch.isclose(norm_across_other_dims_first, norm_across_other_dims_second, atol=1e-6)

# # print(f"Are the computations equivalent? {are_equivalent}")


# # a = torch.randn(1, 2, 3)
# # b = torch.linalg.vector_norm(a, ord=2, dim=1)
# # c = torch.linalg.vector_norm(a, ord=2, dim=(0, 1)).reshape(1,-1)
# # d = 5

# # # Create two tensors with different data types
# # tensor1 = torch.randn(5, dtype=torch.float32)
# # tensor2 = torch.randn(5, dtype=torch.float16)

# # # Attempt to add them
# # result = tensor1 * tensor2
# # b = result.dtype
# # a = 5

# a = torch.randn(4, 5, 5)
# b = torch.tensor([[0, 1],[0, 1]])

# # a = a[]
# # a = torch.randn(4, 5, 5)
# # a_mean = a.mean(axis=0)
# # print('a_mean', a_mean.shape)
# # print('a', a.shape)
# # b = torch.randn(4, 5)
# # e = torch.randn(4, 5)
# # c = torch.matmul(a, b.T)
# # # print(c)
# # c = nn.functional.softmax(c, dim=-1)
# # print('original after softmax', c)
# # c = torch.matmul(c, e)
# # print('new after v', c)

# # a[:, 0] = 0
# # b[:, 0] = 0
# # e[:, 0] = 0
# # d = torch.matmul(a, b.T)
# # # print(d)
# # d = nn.functional.softmax(d, dim=-1)
# # print('new after softmax', d)
# # d = torch.matmul(d, e)
# # print('new after v', d)
# # 假设 a 和 b 是两个三维张量
# # a = torch.rand(2, 3)
# # b = torch.rand(3, 4)

# # c = F.linear(a, b.T)
# # d = torch.tensordot(a, b, dims=[[1]])
# # dd = c == d
# # ddd = d.shape
# # # 在 a 的最后一个维度和 b 的第一个维度上进行点积
# # e = 5
# import torch
# import torch.nn as nn


# import torch

# # Example tensor initialization
# # cos = torch.randn(32, 128, 128)
# # probe_out_dim_indices_for_rope = torch.randint(0, 128, (32, 122))  # Example indices

# # # Create a grid of indices for the batch dimension
# # batch_dim_indices = torch.arange(32).view(-1, 1).expand(-1, 122).to(cos.device)

# # # Use advanced indexing to extract the values
# # extracted_values = cos[batch_dim_indices, :, probe_out_dim_indices_for_rope]

# # # Check the shape of the extracted values
# # print(extracted_values.shape)  # Should be [32, 128, 122]

# import torch

# # Example tensors
# # cos = torch.randn(32, 128, 128)

# # # Assuming position_ids contains indices for the second dimension of cos
# # # and we use every index from 0 to 127 (which is the size of the second dimension of cos)
# # position_ids = torch.arange(128).unsqueeze(0)  # Shape [1, 128]

# # # Perform the indexing operation
# # result = cos[position_ids]

# # # Print the output shape
# # print(result.shape)


# # cos = torch.randn(128, 128)

# # # Assuming position_ids contains indices for the second dimension of cos
# # # and we use every index from 0 to 127 (which is the size of the second dimension of cos)
# # position_ids = torch.arange(128).unsqueeze(0)  # Shape [1, 128]

# # # Perform the indexing operation
# # result = cos[position_ids]

# # # Print the output shape
# # print(result.shape)

# # a = 6

# import torch
# import time

# # Initialize the tensors
# # cos = torch.randn(10, 32, 128, 128)
# # probe_out_dim_indices_for_rope = torch.randint(0, 127, (32, 122))  # Example indices

# # # Operation 1: Loop-based extraction
# # def loop_based_extraction(cos, probe_out_dim_indices_for_rope):
# #     res = cos[:, 0, :, probe_out_dim_indices_for_rope[0]].unsqueeze(1)
# #     for i in range(1, 32):
# #         new = cos[:, i, :, probe_out_dim_indices_for_rope[i]].unsqueeze(1)
# #         res = torch.cat((res, new), dim=1)
# #     return res

# # # Operation 2: torch.gather method
# # def gather_method(cos, probe_out_dim_indices_for_rope):
# #     # Creating index tensor for gather
# #     indices = probe_out_dim_indices_for_rope.unsqueeze(0).unsqueeze(2).expand(10, -1, 128, -1)
# #     # Using torch.gather
# #     gathered = torch.gather(cos, 3, indices)
# #     return gathered



# # # Measure time for loop-based extraction
# # start_time = time.time()
# # result_loop = loop_based_extraction(cos, probe_out_dim_indices_for_rope)
# # end_time = time.time()
# # time_loop = end_time - start_time
# # print(f"Time taken by loop-based extraction: {time_loop} seconds")

# # # Measure time for torch.gather method
# # start_time = time.time()
# # result_gather = gather_method(cos, probe_out_dim_indices_for_rope)
# # end_time = time.time()
# # time_gather = end_time - start_time
# # print(f"Time taken by torch.gather method: {time_gather} seconds")

# # a = 5


# # class SimpleModel(nn.Module):
# #     def __init__(self):
# #         super(SimpleModel, self).__init__()
# #         # self.fc1 = nn.Linear(100, 200)
# #         # self.relu = nn.ReLU()
# #         # self.fc2 = nn.Linear(200, 10)

# #     def forward(self, x):
# #         return torch.matmul(x[0], x[1])
# #         # return x[0] * x[1]

# # # Instantiate the model
# # model = SimpleModel()


# # from deepspeed.profiling.flops_profiler import get_model_profile

# # # Define a batch size and input tensor
# # # batch_size = 32
# # # input_tensor = torch.randn(batch_size, 100)
# # x = torch.randn(2, 100)
# # # y = torch.randn(100)
# # # Use the profiler to get FLOPs and parameter counts
# # flops, macs, params = get_model_profile(model=model, 
# #                                  input_shape=(2,100))
                                

# # print(f"MACs: {macs}")
# # print(f"Parameters: {params}")
# # c = 6
# # # Example instances
# # embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=3)
# # linear_layer = nn.Linear(in_features=10, out_features=5)

# # # Check if each instance is an instance of Embedding or Linear
# # print(isinstance(embedding_layer, nn.Embedding))  # True
# # print(isinstance(embedding_layer, nn.Linear))     # False

# # print(isinstance(linear_layer, nn.Embedding))     # False
# # print(isinstance(linear_layer, nn.Linear))        # True
# # self.exclude_dim_to_aggregate = None
# # self.sort_norm_dim = 0
# # a = torch.tensor([5])

# # # Add an extra dimension
# # # To add it at the 0th dimension (making it 1x1)
# # a = a.unsqueeze(0)

# # # Now a is a 2D tensor with shape (1, 1)
# # print(a)  # Outputs tensor([[5]])
# # print(a.shape)  # Outputs torch.Size([1, 1])
# # a = 'lora'

# # b = a.split('-')

# # print(b)
# # sub_dense_info = ['', 0, 3.74174427986145, 124647170, 0,'zzzz']
# # NUM_PARAMETER_UNIT = (1000000, 'Million')
# # FLOPS_UNIT = (1000000, 'Million')
# # # already in seconds unit
# # TIME_UNIT = (1, 's')
# # print(FLOPS_UNIT[0])
# # print(f"dense: {sub_dense_info[0]} - {sub_dense_info[1]/FLOPS_UNIT[0]:.2f}" , flush=True)

# # print(f"dense: {sub_dense_info[0]} - {sub_dense_info[1]/{FLOPS_UNIT[0]}:.2f} {FLOPS_UNIT[1]}Flops - {sub_dense_info[2]/TIME_UNIT[0]:.2f} {TIME_UNIT[1]} \
# #               - {sub_dense_info[3]/NUM_PARAMETER_UNIT[0]:.2f} {NUM_PARAMETER_UNIT[1]} parameters - {sub_dense_info[4]}", flush=True)

# # a = torch.tensor([1,2,3,4])

# # print(a.numel())

# # b = a[0:0]
# # print(b)
# # print(b.numel())
# import copy
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # input_data = np.array([0.1 for i in range(100053)])

# # # input_data_length = len(input_data)
# # compress_ratio = 1000
# # pace = int(len(input_data) // compress_ratio)
# # simplified_input_data = np.array([input_data[i] for i in range(0, len(input_data), pace)])
# # x = np.array(list(range(len(simplified_input_data)+1)))
# # y = np.array(list(range(len(simplified_input_data)+1)))
# # x, y = np.meshgrid(x, y)
# # eta = np.full(x.shape, np.nan)
# # # eta = np.full(x.shape, 6, dtype=float)
# # # mask = y < x
# # print('eta', eta.shape)
# # mask = y < x

# # # Applying the mask
# # x = np.where(mask, x, np.nan)  # Replace values not in the upper triangle with NaN
# # y = np.where(mask, y, np.nan)

# # pq_p = 1
# # pq_q = 2

# import torchvision.models as models
# import torch


# # def new_forward(self, x):
# #     x = self.conv1(x)
# #     x = self.bn1(x)
# #     x = self.relu(x)
# #     x = self.maxpool(x)

# #     x = self.layer1(x)
# #     x = self.layer2(x)
# #     x = self.layer3(x)
# #     x = self.layer4(x)
# #     return x


# # # define a resnet instance
# # resnet = models.resnet18()

# # # add new_forward function to the resnet instance as a class method
# # bound_method = new_forward.__get__(resnet, resnet.__class__)
# # setattr(resnet, 'forward', bound_method)
# # aa = resnet.forward
# a = 5
# # # print(len(x), len(x[0]))
# # for d in range(1, len(x)):
# #     # m at most equals to d-1
# #     cur_dimension = d * pace
# #     pq_index = simplified_input_data[d-1]
# #     for m in range(1, d):
# #     # for m in range(1, len(x[0])):
# #         cur_rest_dimension = m * pace

# #         sub_eta = ((cur_rest_dimension / (((1 - pq_index) ** (pq_q * pq_p / (pq_q - pq_p))) * cur_dimension)) ** (-(pq_q - pq_p) / pq_q)) - 1
# #         # print('sub_eta', sub_eta)
# #         # print(d, m, sub_eta)
# #         if sub_eta < -1:
# #             sub_eta = -1
# #         elif sub_eta > 2:
# #             sub_eta = 2
# #         # print(type(sub_eta))

# #         # print('d', d, 'm', m, 'sub_eta', sub_eta, type(d), type(m))
# #         eta[m][d] = sub_eta

# # # for i in range(1, len(x[0])):
# # #     # fix = i
# # #     # for j in range(1, fix):
# # #     for j in range(1, len(x[0])):
# # #         eta[i][j] = 3

# # z = np.sin(np.sqrt(x**2 + y**2))
# # print('z', z.shape)
# # # Create a figure and a 3D axis
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # print('eta', eta)
# # # Plot a 3D surface
# # surf = ax.plot_surface(x, y, eta, cmap='viridis')
# # # surf = ax.plot_surface(x, y, z, cmap='viridis')
# # # Add a color bar which maps values to colors
# # fig.colorbar(surf, shrink=0.5, aspect=5)

# # ax.set_title('3D Heatmap')
# # ax.set_xlabel('d dimension')
# # ax.set_ylabel('m dimension')
# # ax.set_zlabel('eta')
# # plt.show()

# a = 5
# # a = [[0.40625],
# #  [0.4375 ]]

# # b = np.std(a, axis=1)
# # c = np.std(a, axis=0)
# # d = 5
# # c = torch.empty(0)
# # d = torch.empty(3, 0, 2)

# # print(c, c.numel())
# # print(d, d.numel())

# # a = torch.tensor([[1,2,5,9,10], [1,3, 8, 15, 20]], dtype=torch.float32)
# # standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)
# # b = a ** 2
# # c = standarlization(a)
# # d = standarlization(b)
# # e = 5
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


# import torch
# import psutil
# import os

# def memory_usage_in_MB():
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / (1024 * 1024)  # Memory in MB

# # Memory usage before the operation


# # Your PyTorch code
# # a = torch.randn(40, 1100)
# # b = torch.randn(1100, 40)

# # temp_a = copy.deepcopy(a)
# # temp_b = copy.deepcopy(b)
# # memory_before = memory_usage_in_MB()
# # c = torch.matmul(a, b)

# # # Memory usage after the operation
# # memory_after = memory_usage_in_MB()

# # # Calculate the difference
# # memory_consumed = memory_after - memory_before
# # print(f"Memory consumed: {memory_consumed:.2f} MB")


# # memory_before = memory_usage_in_MB()
# # a = a.unsqueeze(-1)
# # b = b.unsqueeze(0)
# # result = a * b
# # print('result', result.shape)
# # memory_after = memory_usage_in_MB()

# # # Calculate the difference
# # memory_consumed = memory_after - memory_before
# # print(f"Memory consumed broadcast: {memory_consumed:.2f} MB")


# # memory_before = memory_usage_in_MB()
# # result = (a * b).sum(dim=(0,2))
# # print('result', result.shape)

# # # result2 = a.sum(dim=0) * b.sum(dim=1)
# # # print('result2', result2.shape)

# # result3 = temp_a.sum(0) * temp_b.sum(1)
# # print('result3', result3.shape)
# # dd = result == result3

# # result4 = (temp_a * temp_b.sum(1)).sum(0)
# # ddd = result == result4
# # memory_after = memory_usage_in_MB()

# # # Calculate the difference
# # memory_consumed = memory_after - memory_before
# # print(f"Memory consumed 2: {memory_consumed:.2f} MB")


# # b = torch.randn(256, 128)
# # c = a * b
# # d = 6
# # Settings for the network
# # layer_sizes = [5, 8, 8, 5]  # Example layer sizes
# # n_layers = len(layer_sizes)
# # layer_positions = range(n_layers)
# # node_radius = 0.05

# # # Create the plot
# # fig, ax = plt.subplots()

# # # Function to draw nodes
# # def draw_layer(y, size, label):
# #     x_values = [x * 0.2 for x in range(size)]
# #     for x in x_values:
# #         circle = plt.Circle((x, y), node_radius, color='blue', fill=True)
# #         ax.add_artist(circle)
# #     # Optionally add a label for the layer
# #     ax.text(x_values[-1] + 0.15, y, label, fontsize=12)

# # # Draw the layers
# # for i, size in enumerate(layer_sizes):
# #     draw_layer(layer_positions[i], size, f'Layer {i+1}')

# # # Highlight the middle layer
# # middle_layer_index = n_layers // 2 - 1
# # highlight_rect = patches.Rectangle((-0.1, middle_layer_index - 0.1), 
# #                                    layer_sizes[middle_layer_index] * 0.2, 
# #                                    0.3, linewidth=2, edgecolor='r', facecolor='none')
# # ax.add_patch(highlight_rect)

# # # Annotate
# # ax.annotate('Our theory-guided adaptive pruning', 
# #             xy=(layer_sizes[middle_layer_index] * 0.1, middle_layer_index), 
# #             xytext=(layer_sizes[middle_layer_index] * 0.5, middle_layer_index + 1),
# #             arrowprops=dict(facecolor='black', shrink=0.05))

# # # Set the limits and labels
# # ax.set_xlim(-0.2, max(layer_sizes) * 0.2)
# # ax.set_ylim(-1, n_layers)
# # ax.set_aspect('equal', adjustable='datalim')
# # ax.axis('off')
# a = torch.tensor([[0.6815, 0.7796, 0.9360, 0.5866, 1.8860],
#  [0.1141, 0.1273, 0.4898, 1.0005, 0.2570],
#  [0.2012, 0.2757, 0.2001, 1.2834, 0.4445]])
# b = a / torch.sum(a, axis=-1, keepdim=True)
# print('a', a, 'b', b)
# # class Fulei:
# #     def __init__(self):
# #         pass

# #     def fuleicall(self):
# #         print(self.weight)

# # class zilei(Fulei):
# #     def __init__(self):
# #         super().__init__()
# #         self.weight = 16

# #     def zileicall(self):
# #         self.fuleicall()


# # a = zilei()
# # a.zileicall()
# # b = 5

# # a = torch.tensor([[1, 2, 3, 4, 5], [6,7,8,9,10]])

# # b = torch.tensor([[11, 12, 13], [16,17,18]])

# # a[..., [1,2,3]] = b

# # print(a[0], a[1])
# # c = 5
# # plt.show()

# # import torch

# # # Define dimensions
# # C_out = 3  # Number of output channels
# # C_in = 4   # Number of input channels
# # N = 2      # Number of samples
# # L = 1      # Additional factor (for simplicity, we keep it 1)

# # # Define desired sparsity
# # s = 0.5  # 50% sparsity

# # # Create a random weight matrix W and input matrix X
# # W = torch.randn(C_out, C_in)
# # X = torch.randn(N * L, C_in)

# # # Define the pruning function with the correction
# # def prune(W, X, s):
# #     temp = X.norm(p=2, dim=0)
# #     print('temp', temp)
# #     metric = W.abs() * X.norm(p=2, dim=0)
# #     print('metric', metric)
# #     _, sorted_idx = torch.sort(metric, dim=1)
# #     print('sorted_idx', sorted_idx)
# #     pruned_idx = sorted_idx[:, :int(C_in * s)]
# #     print('pruned_idx', pruned_idx)
# #     # Create a tensor of zeros with the same shape as the pruned indices
# #     zeros = torch.zeros_like(W[:, :int(C_in * s)])
# #     W.scatter_(dim=1, index=pruned_idx, src=zeros)
# #     return W

# # # Apply the pruning function
# # W_pruned = prune(W, X, s)

# # import numpy as np

# # # Define custom bin edges
# # bin_edges = [
# #     -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, # -1000 to -100
# #     -50, 0, 50, 100,  # -100 to 100
# #     200, 300, 400, 500, 600, 700, 800, 900, 1000  # 100 to 1000
# # ]

# # # Fine bins around -10 to 10
# # fine_bins = np.arange(-10, 10, 0.1).tolist()
# # bin_edges = bin_edges + fine_bins + [1e-3]

# # # Sort the bin edges
# # bin_edges = sorted(set(bin_edges))
# # print(bin_edges)
# # # Example data from a batch
# # batch_data = np.random.uniform(-1000, 1000, 1000)  # Replace with your actual batch data

# # # Bin the data
# # hist, _ = np.histogram(batch_data, bins=bin_edges)

# # # hist now contains the count of data points in each bin
# # print(hist)


# # import matplotlib.pyplot as plt

# # # Example histogram data
# # hist_data = [10, 15, 7, 12, 5]  # Frequency counts for each bin

# # # Corresponding bin edges
# # bin_edges = [0, 1, 2, 3, 4, 5]  # Define the range of each bin

# # # Calculate the width for each bin
# # bin_widths = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges)-1)]

# # print(bin_edges[:-1])
# # # Create the bar plot
# # plt.bar(bin_edges[:-1], hist_data, width=bin_widths, align='edge')

# # # Labeling
# # plt.xlabel('Value Range')
# # plt.ylabel('Frequency')
# # plt.title('Histogram')

# # # Show the plot
# # plt.show()



# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Generate sample data
# # data = np.random.normal(0, 1, 1000)

# # # Compute histogram
# # counts, bins = np.histogram(data, bins=30)
# # counts = [100, 100, 100]
# # bins = [-0.1, 0, 0.1, 0.2]
# # a = (sum(counts) * np.diff(bins))
# # print('a', a)
# # print('counts', counts)
# # print('sum(counts)', sum(counts))
# # print('bins', bins),
# # print('np.diff(bins)', np.diff(bins)),

# # # Calculate density
# # density = counts / (sum(counts) * np.diff(bins))
# # print('density', density)
# # # Plotting the histogram as a density
# # # plt.bar(bins[:-1], density, width=np.diff(bins), edgecolor='black')
# # plt.hist(data, bins, density=True, edgecolor='black')

# # plt.title('Density Histogram')
# # plt.xlabel('Value')
# # plt.ylabel('Density')

# # plt.show()

# import copy
# from datasets import load_dataset
# from transformers import AutoTokenizer

# # for x in range(2, 30, 10):
# #     print(x)
# # eta = 0
# # pq_p = 1
# # pq_q = 2
# # prune_norm = 2
# # beta = 0.9
# # gamma = 1
# import time
# # class YourClass:
# #     # Other methods...
# #     def __init__(self):
# #         self.logger_info_time_used = 0

# #     def monitor_time(func):
# #         def wrapper(*args, **kwargs):
# #             print('wrapper', args, kwargs)
# #             start_time = time.time()
# #             result = func(*args, **kwargs)
# #             args[0].logger_info_time_used += time.time() - start_time
# #             return result
# #         return wrapper
    
# #     @monitor_time
# #     def update_pruning_info(self, info):
# #         a = 5

# # a = YourClass()
# # a.update_pruning_info(1)
# # def calculate_entropy(probabilities):
# #     """
# #     Calculate the entropy of a probability distribution.
# #     :param probabilities: list of probabilities for each event
# #     :return: entropy of the distribution
# #     """
# #     entropy = 0
# #     for p in probabilities:
# #         if p > 0:  # To avoid math domain error for log(0)
# #             entropy -= p * math.log(p, 2)  # Log base 2 for entropy in bits
# #     return entropy

# # def pq_struct(w, key, prune_dim):

# #     calc_dim = 1
# #     # i != prune_dim and 
# #     # dims_to_aggregate = tuple(i for i in range(w.dim()) if i != 0)
# #     # norm_across_other_dims = torch.linalg.vector_norm(w, ord=prune_norm, dim=dims_to_aggregate)     
# #     print(w)
# #     norm_p = torch.linalg.vector_norm(w, ord=pq_p, dim=calc_dim)
# #     norm_q = torch.linalg.vector_norm(w, ord=pq_p, dim=calc_dim) + 1e-10
    
# #     dimension = w.shape[prune_dim]
# #     pq_indices = (1 - dimension ** (1/pq_q - 1/pq_p) * norm_p / norm_q)

# #     # add additional dimension if dimension is 0
# #     if pq_indices.dim() == 0:
# #         pq_indices = pq_indices.unsqueeze(0)

# #     if torch.isnan(pq_indices).any():
# #         raise ValueError('pq_indices contains nan values')

# #     lower_bound = dimension * (1 + eta) ** (-pq_q / (pq_q - pq_p)) * (1 - pq_indices) ** (pq_q * pq_p / (pq_q - pq_p))
# #     beta_tensor = torch.full_like(lower_bound, beta)
# #     prune_channels_count = torch.floor(dimension * torch.min(gamma * (1 - lower_bound / dimension), beta_tensor))

# #     _, sorted_channels = torch.sort(norm_across_other_dims, dim=calc_dim)
# #     prune_channels = sorted_channels[:int(prune_channels_count.item())]
# #     # info = {
# #     #     f"{key}_norm_across_other_dims": norm_across_other_dims.mean(dim=0).squeeze(0).tolist(),
# #     #     f"{key}_pq_indices": pq_indices.mean(dim=0).squeeze(0).tolist(),
# #     # }
# #     # self.update_pruning_info(info)
# #     return prune_channels

# # tensor1 = torch.tensor([1, 1, 1, 1, 1, 10, 10, 10, 10, 10])
# # tensor2 = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 100])
# # # sum_tensor2 = tensor1.sum()
# # import math

# # # Now you can use log from the math module
# # print(-math.log(0.1))

# # Normalize tensor2 by dividing each element by the sum
# # normalized_tensor2 = tensor1 / sum_tensor2
# # Stack the tensors to create a batched tensor
# # The resulting tensor will have shape [2, 10]
# # batched_tensor = torch.stack([tensor1, tensor2])
# # batched_tensor = torch.stack([tensor1, normalized_tensor2])
# # pq_struct(batched_tensor, 'w', 1)

# import numpy as np

# save_format = 'png'
# fig_name = 'zz'
# # Define the base directory for visualization
# vis_path = './output/vis/{}'.format(save_format)

# # Construct the full path for the figure
# fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
# print(vis_path, fig_path)
# # Create a sample matrix X (e.g., 4 samples, 3 features)
# # X = np.array([[1, 2, 3],
# #               [4, 5, 7],
# #               [7, 9, 9],
# #               [10, 16, 12]])

# # # Compute X^T X
# # XTX = np.dot(X.T, X)

# # # Approach 1: Full matrix inversion of X^T X, then extract diagonal
# # inv_XTX = np.linalg.inv(XTX)
# # diag_inv_XTX = np.diag(inv_XTX)

# # # Approach 2: Extract the diagonal of X^T X, then invert each element
# # diag_XTX = np.diag(XTX)
# # inv_diag_XTX = 1 / diag_XTX

# # # Display the results
# # print("Diagonal of (X^T X)^-1:\n", diag_inv_XTX)
# # print("Inverse of the diagonal elements of X^T X:\n", inv_diag_XTX)

# a = 5
# # a = 5
# # def preprocess_function_test(dataset):
# #     all_text = "\n\n".join(dataset['text'])
# #     model_inputs = tokenizer(all_text, return_tensors='pt', truncation=False, padding=False)

# #     max_length = 512  # Set your desired max length
# #     input_ids = model_inputs['input_ids'][0]  # Assuming a single concatenated string
# #     attention_mask = model_inputs['attention_mask'][0]

# #     input_chunks = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length)]
# #     mask_chunks = [attention_mask[i:i + max_length] for i in range(0, len(attention_mask), max_length)]

# #     final_inputs = []
# #     for chunk in input_chunks:
# #         final_inputs.append({
# #             'input_ids': torch.tensor(chunk, dtype=torch.long),
# #             'attention_mask': torch.tensor(mask_chunks[final_inputs.index(chunk)], dtype=torch.long)
# #         })

# #     # Add labels if required
# #     for item in final_inputs:
# #         item['labels'] = item['input_ids'].clone()

# #     return final_inputs

# # def load_and_tokenize_dataset(model_checkpoint, dataset_name='wikitext', dataset_version='wikitext-2-v1', max_length=512):
# #     # count = 0
# #     # Load the dataset
# #     dataset = load_dataset(dataset_name, dataset_version, split='test')

# #     a = dataset['text']
# #     # Load the tokenizer    
# #     tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# #     if tokenizer.pad_token_id is None:
# #         tokenizer.pad_token_id = tokenizer.eos_token_id
#     # Tokenization function
#     # def preprocess_function_test(examples):
#     #     max_length = 120
#     #     text_column = ['text']
#     #     batch_size = len(examples[text_column[0]])
        
#     #     model_inputs = tokenizer("\n\n".join(examples['text']), return_tensors='pt')
#     #     labels = tokenizer("\n\n".join(examples['text']), return_tensors='pt')
#     #     nsamples = model_inputs["input_ids"].numel() // max_length

#     #     for i in range(nsamples):
#     #         start = i * max_length
#     #         end = start + max_length
#     #         sample_input_ids = model_inputs["input_ids"][:, start:end]
#     #         sample_attention_mask = model_inputs["attention_mask"][:, start:end]
#     #         # label_input_ids = labels[i]
#     #         sample_input_ids.reshape(1, max_length)
#     #         sample_attention_mask.reshape(1, max_length)
#     #         model_inputs["input_ids"][i] = sample_input_ids
#     #         model_inputs["attention_mask"][i] = sample_attention_mask
#     #         # labels["input_ids"][i] = label_input_ids
#     #         # model_inputs["split"].append(cfg['task_label'][examples['category'][i]])
#     #         # model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][-max_length:])
#     #         # model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][-max_length:])
#     #         labels["input_ids"][i] = sample_input_ids
#     #         labels["attention_mask"][i] = sample_attention_mask
#     #     model_inputs["labels"] = labels["input_ids"]


#     #         # Tokenize all examples
#     #     model_inputs = tokenizer("\n\n".join(examples['text']), max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
#     #     a = model_inputs["input_ids"].numel()
#     #     # In this example, labels are the same as the input. Modify as needed.
#     #     labels = tokenizer("\n\n".join(examples['text']), max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

#     #     # The input_ids and attention_mask are already in the desired format
#     #     model_inputs["labels"] = labels["input_ids"]

#     #     return model_inputs

#     # def remove_empty_examples(example):
#     #     return example["text"].strip() != ""

#     # model_inputs = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
#     # labels = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
#     # nsamples = model_inputs["input_ids"].numel() // max_length
#     # for i in range(nsamples):
#     #     start = i * max_length
#     #     end = start + max_length
#     #     sample_input_ids = model_inputs["input_ids"][:, start:end]
#     #     sample_attention_mask = model_inputs["attention_mask"][:, start:end]
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



# load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], split='validation')



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

# # a = torch.randn((3, 5))
# # # torch.sum((torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1))).reshape(-1, 1, 1) * torch.linalg.vector_norm(subset[name].weight.data, ord=1, dim=1)), dim=1)
# # b = torch.linalg.vector_norm(a, ord=1, dim=1)
# c = 5
# # def compare_norms_multi_dim(tensor, dims):
# #     # Calculate the norm using .norm()
# #     norm_result = tensor.norm(dim=dims)

# #     # Calculate the norm using torch.linalg.vector_norm()
# #     vector_norm_result = torch.linalg.vector_norm(tensor, dim=dims)

# #     # Output comparison
# #     print(f"Tensor shape: {tensor.shape}, Dimensions: {dims}")
# #     print("Result using .norm():", norm_result)
# #     print("Result using torch.linalg.vector_norm():", vector_norm_result)
# #     print("Are the results the same?:", torch.allclose(norm_result, vector_norm_result))
# #     print("\n")

# # # Create a 4D tensor with random values
# # tensor = torch.randn(4, 4, 4, 4)

# # # Compare norms across dimensions (2, 3)
# # compare_norms_multi_dim(tensor, dims=(2, 3))
# # compare_norms_multi_dim(tensor, dims=(1, 3))
# # compare_norms_multi_dim(tensor, dims=(1,2, 3))


# # tensor = torch.tensor([4,5,6,7]) 
# # a0 = tensor.dim()
# # a = tensor.shape
# # # a1 = a[0]
# # b = tensor.unsqueeze(0)
# # c = b.shape
# # c = 5


# import torch
# from torch.nn import CrossEntropyLoss


# a = nn.Parameter(torch.randn(3, 4))
# print(a.dim(), a.shape)
# a.data = torch.randn(3, 4)
# print(a)
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


