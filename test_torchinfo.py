import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 16 * 16, 128)  # Assuming input size is 32x32
        self.fc2 = nn.Linear(128, 10)  # 10 classes for example


        self.fc1.weight.data = self.fc1.weight.data[:, :16*16]
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x_temp = x + 1 + 1 + 1 + 1 + 1
        x = x[:, :16*16]
        
        x = F.relu(self.fc1(x))

        
        # x = x[:, :12]
        x = self.fc2(x)
        return x
    

from torchinfo import summary

model = SimpleCNN()
# input = torch.randn(1, 3, 32, 32)

summary(model, input_size=(10, 3, 32, 32))