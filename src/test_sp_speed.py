import torch
import torch.nn as nn
import time

class CustomLinearModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=20):
        super(CustomLinearModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_size, output_size))
            # Adjust input size for the next layer if needed (for pruning logic)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def prune_model(model, prune_percentage=0.1):
    for i in range(len(model.layers)):
        if i % 2 == 0:  # Even-indexed layers
            # Prune 10% of the output dimensions
            # For simplicity in demonstration, we're adjusting the weight size directly
            # In practice, you'd use structured pruning methods from torch.nn.utils.prune
            output_features = model.layers[i].out_features
            new_out_features = int(output_features * (1 - prune_percentage))
            model.layers[i] = nn.Linear(model.layers[i].in_features, new_out_features)
        else:
            # Adjust the input dimension of the odd layers to match the pruned output of the even layers
            input_features = model.layers[i].in_features
            new_in_features = int(input_features * (1 - prune_percentage))
            model.layers[i] = nn.Linear(new_in_features, model.layers[i].out_features)

# Parameters
input_size = 4096
output_size = 4096
num_layers = 20
batch_size = 10
n_batches = 50

# Initialize model
model = CustomLinearModel(input_size, output_size, num_layers)

# Generate dummy data
inputs = [torch.randn(batch_size, input_size) for _ in range(n_batches)]

# Measure runtime without pruning
start_time = time.time()
with torch.no_grad():
    for input in inputs:
        output = model(input)
end_time = time.time()
runtime_without_pruning = end_time - start_time

# Prune the model
prune_model(model, 0.1)

# Measure runtime with pruning
start_time = time.time()
with torch.no_grad():
    for input in inputs:
        output = model(input)
end_time = time.time()
runtime_with_pruning_10 = end_time - start_time

# Prune the model
prune_model(model, 0.1)

# Measure runtime with pruning
start_time = time.time()
with torch.no_grad():
    for input in inputs:
        output = model(input)
end_time = time.time()
runtime_with_pruning_20 = end_time - start_time

# Prune the model
prune_model(model, 0.1)

# Measure runtime with pruning
start_time = time.time()
with torch.no_grad():
    for input in inputs:
        output = model(input)
end_time = time.time()
runtime_with_pruning_30 = end_time - start_time

# Prune the model
prune_model(model, 0.1)

# Measure runtime with pruning
start_time = time.time()
with torch.no_grad():
    for input in inputs:
        output = model(input)
end_time = time.time()
runtime_with_pruning_40 = end_time - start_time

print("Runtime without pruning: {:.4f} seconds".format(runtime_without_pruning))
print("Runtime with 10% pruning: {:.4f} seconds".format(runtime_with_pruning_10))
print("Runtime with 20% pruning: {:.4f} seconds".format(runtime_with_pruning_20))
print("Runtime with 30% pruning: {:.4f} seconds".format(runtime_with_pruning_30))
print("Runtime with 40% pruning: {:.4f} seconds".format(runtime_with_pruning_40))

