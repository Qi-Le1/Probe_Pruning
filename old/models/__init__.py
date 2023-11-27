import torch.nn as nn
from .wresnet import (
    WideResNet,
    wresnet28x2,
    wresnet28x8,
    wresnet37x2
)

from .resnet import (
    ResNet,
    resnet9,
    resnet18
)

# from .cnn import create_CNN

from .generator import Generator


class cnn(nn.Module):
    def __init__(self, input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10):
        super(SimpleCNN, self).__init__()
        # self.n1 = nn.GroupNorm(1, input_dim)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.n1 = nn.GroupNorm(1, 6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.n2 = nn.GroupNorm(1, 16)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
    
    def f(self, x, start_layer_idx):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.n1(x)
        # # print('normalize', flush=True)
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.n2(x)
        # x = x.view(-1, 16 * 5 * 5)
        if start_layer_idx == -1:
            return self.fc3(x)
        
        x = self.n1(self.conv1(x))
        x = self.pool(F.relu(x))
        # print('normalize', flush=True)
        x = self.n2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x
        
    def forward(self, input, start_layer_idx=None):
        if start_layer_idx == -1:
            output = {}
            output['target'] = self.f(input['data'], start_layer_idx)
            return output
        
        output = {}
        output['target'] = self.f(input['data'], start_layer_idx)
        output['loss'] = loss_fn(output['target'], input['target'])
        return output