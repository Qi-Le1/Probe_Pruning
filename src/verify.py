# import argparse
# import datetime
# import os
# import shutil
# import time
# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
# from collections import defaultdict
# from config import cfg, process_args
# from dataset import make_dataset, make_data_loader, process_dataset, collate
# from metric import make_metric, make_logger
# from model import make_model, make_optimizer, make_scheduler, make_ft_model, freeze_model, make_cola
# from module import save, to_device, process_control, resume, makedir_exist_ok, PeftModel
import torch.backends.cudnn as cudnn
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import argparse
from functools import reduce
from model.eri import EriModel
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import itertools
from deepspeed.profiling.flops_profiler import FlopsProfiler
import numpy as np
from config import cfg, process_args
from module import save, to_device, process_control, resume, makedir_exist_ok

'''
verity pruning method and FLOPs calculation
'''

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
mil = MB

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


class FCBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class ModularNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ModularNN, self).__init__()
        self.block1 = FCBlock(input_size, hidden_size)
        self.block2 = FCBlock(hidden_size, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x
        
class SimpleDataset(Dataset):
    def __init__(self, num_samples, input_size, num_classes):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random sample
        sample = torch.randn(int(self.input_size/2), self.input_size)
        # Generate a random label
        label = torch.randint(0, self.num_classes, (1,))
        return sample

def print_info(prof):
    for name, module in prof.model.named_modules():
        print('name', name)
        print('module.__flops__', module.__flops__)


def vanilla_model_inference(model, input):
    model_prof = FlopsProfiler(model)
    model_prof.start_profile()
    example_output = model_prof.model(input)
    model_prof.stop_profile()
    print_info(model_prof)
    return

def modified_model_inference(model, input):
    model_prof = FlopsProfiler(model)
    model_prof.start_profile()
    example_output = model_prof.model(input)
    model_prof.stop_profile()
    print_info(model_prof)
    return


def verify_eri():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        
    input_size = 10  # Number of input features
    hidden_size = 5  # Number of neurons in the hidden layer
    num_classes = 3  # Number of output classes
    model = ModularNN(input_size, hidden_size, num_classes)

    # Define the dataset
    num_samples = 100  # Total number of samples in the dataset
    dataset = SimpleDataset(num_samples, input_size, num_classes)

    # Create the DataLoader
    batch_size = 1  # Number of samples per batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, input in enumerate(dataloader):
        print('input', input)
        vanilla_model_inference(model, input)
        model = EriModel(model)
        modified_model_inference(model, input)
        break
def verify_address():
    # tensor = torch.randn(3, 4)
    # address = tensor.data_ptr()
    # print('address', address)   
    # temp_a = torch.index_select(tensor, 0, torch.tensor([0, 1]))
    # address_a = temp_a.data_ptr()
    # print('address_a', address_a)

    # a = torch.randn(3, 4)
    # b = a[1:, :]
    # print(a.data_ptr(), b.data_ptr())

    # mask = a > 0
    # c = a[mask]
    # print(a.data_ptr(), c.data_ptr())

    large_tensor = torch.randn(1000, 100, 50, 50)

    # Select indices
    indices = torch.randint(0, 100, (50,))

    # Measure time for torch.index_select
    start_time = time.time()
    selected_tensor_torch = torch.index_select(large_tensor, 1, indices)
    torch_duration = time.time() - start_time

    # Create a mask of the same shape as large_tensor
    # Initialize the mask as False (not selected)
    mask = torch.zeros_like(large_tensor, dtype=torch.bool)

    # Update the mask for selected indices (set True for selected indices)
    # We'll set the mask to True only along the second dimension using broadcasting
    mask[:, indices, :, :] = True

    # Measure time for indexing with a mask
    start_time = time.time()
    selected_tensor_mask = large_tensor[mask]
    mask_duration = time.time() - start_time

    print("Duration using a mask:", mask_duration)

    # Convert to NumPy, perform index_select, and convert back
    large_tensor_numpy = large_tensor.numpy()
    indices_numpy = indices.numpy()

    start_time = time.time()
    selected_tensor_numpy = torch.from_numpy(large_tensor_numpy[:, indices_numpy, :, :])
    numpy_duration = time.time() - start_time

    print("Duration using PyTorch:", torch_duration)
    print("Duration using NumPy:", numpy_duration)


    # print(f'{name} - {module.__flops__/KB:.2f} KFlops - {module.__duration__*Sec:.2f} ms - {module.__params__/mil:.2f} mil - {module.__macs__/KB:.2f} KMacs - {type(module)}')
if __name__ == "__main__":
    # verify_gl()
    # verify_cola()
    verify_eri()
    # verify_address()