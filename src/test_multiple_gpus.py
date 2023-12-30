
import argparse
import os
import time
import copy
import time
import random
import torch
import traceback
import datetime
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate, make_batchnorm_stats
from metric import make_metric, make_logger
from model import make_model, make_prune_model
from module import save, to_device, process_control, resume, makedir_exist_ok, \
    record_pruing_info, get_model_profile, summarize_info_list, match_prefix
from deepspeed.profiling.flops_profiler import FlopsProfiler



cudnn.benchmark = True
def main():
    import torch
    import torch.nn as nn

    class MyModel(nn.Module):
        def __init__(self, gpu_ids):
            super(MyModel, self).__init__()
            # First half of the model
            self.part1 = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 30)
            ).to(gpu_ids[0])

            # Second half of the model
            self.part2 = nn.Sequential(
                nn.Linear(30, 20),
                nn.ReLU(),
                nn.Linear(20, 10)
            ).to(gpu_ids[1])

        def forward(self, x):
            # Process input on GPU 0
            x = self.part1(x.to(gpu_ids[0]))
            # Move intermediate output to GPU 1 and process the second part
            x = self.part2(x.to(gpu_ids[1]))

            return x

    def get_available_gpus():
        """
        Returns a list of available GPU IDs.
        """
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_ids = list(range(num_gpus))
            gpu_names = [torch.cuda.get_device_name(i) for i in gpu_ids]
            return gpu_ids, gpu_names
        else:
            return [], []

    gpu_ids, gpu_names = get_available_gpus()
    print("Available GPUs:")
    for gpu_id, gpu_name in zip(gpu_ids, gpu_names):
        print(f"ID: {gpu_id}, Name: {gpu_name}")

    # Initialize the model
    model = MyModel(gpu_ids)

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Dummy dataset
    inputs = torch.randn(64, 10)  # Batch size of 64, 10 features
    targets = torch.randint(0, 10, (64,))  # Random targets

    # Training loop
    for epoch in range(2):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.to(gpu_ids[-1]), targets.to(gpu_ids[-1]))  # Move targets to GPU 1

        # Backward and optimize
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    







if __name__ == '__main__':
    main()