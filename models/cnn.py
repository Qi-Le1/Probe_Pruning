import torch.nn as nn
import torch.nn.functional as F
# from ..utils.api import CONFIGS_
from .utils import init_param, make_batchnorm, loss_fn, CustomReLU, InferenceConv2d, InferenceLinear
from config import cfg
import collections

#################################
##### Neural Network model #####
#################################
class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=62, relu_threshold=0):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.n1 = nn.GroupNorm(1, 6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.n2 = nn.GroupNorm(1, 16)
        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

        self.relu1 = CustomReLU(relu_threshold)
        self.relu2 = CustomReLU(relu_threshold)
        self.relu3 = CustomReLU(relu_threshold)
        self.relu4 = CustomReLU(relu_threshold)

    def f(self, x, start_layer_idx):

        if start_layer_idx == -1:
            return self.fc3(x)
        
        x = self.n1(self.conv1(x))
        # x = self.pool(F.relu(x))
        x = self.pool(self.relu1(x))
        # print('normalize', flush=True)
        x = self.n2(self.conv2(x))
        # x = self.pool(F.relu(x))
        x = self.pool(self.relu2(x))
        x = x.view(-1, 16 * 4 * 4)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        # print('fc2 output shape', x.shape, flush=True)
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

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10, relu_threshold=0):
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

        self.relu1 = CustomReLU(relu_threshold)
        self.relu2 = CustomReLU(relu_threshold)
        self.relu3 = CustomReLU(relu_threshold)
        self.relu4 = CustomReLU(relu_threshold)

        # if 'current_mode' in cfg and cfg['current_mode'] == 'test':
        #     self.conv1 = InferenceConv2d(self.conv1)
        #     self.conv2 = InferenceConv2d(self.conv2)
            
        #     self.fc1 = InferenceLinear(self.fc1)
        #     self.fc2 = InferenceLinear(self.fc2)
        #     self.fc3 = InferenceLinear(self.fc3)

    def f(self, x, start_layer_idx):

        if start_layer_idx == -1:
            return self.fc3(x)
        a = 'current_mode' in cfg
        if 'current_mode' not in cfg or cfg['current_mode'] == 'train':
            x = self.n1(self.conv1(x))
            # x = self.pool(F.relu(x))
            x = self.pool(self.relu1(x))
            # print('normalize', flush=True)
            x = self.n2(self.conv2(x))
            # x = self.pool(F.relu(x))
            x = self.pool(self.relu2(x))
            x = x.view(-1, 16 * 5 * 5)

            # x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.fc2(x))
            x = self.fc3(x)
        elif 'current_mode' in cfg and cfg['current_mode'] == 'test':
            x = self.n1(self.conv1(x))
            # x = self.pool(F.relu(x))
            x, selected_channel = self.relu1(x)
            x = self.pool(x)
            # print('normalize', flush=True)
            x = self.n2(self.conv2(x, selected_channel))
            # x = self.pool(F.relu(x))
            x, selected_channel = self.relu2(x)
            x = self.pool(x)
            x = x.view(-1, 16 * 5 * 5)

            # x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            x, selected_channel = self.relu3(self.fc1(x))
            x, selected_channel = self.relu4(self.fc2(x, selected_channel))
            x = self.fc3(x, selected_channel)
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

def create_CNN():
    model = None
    target_size = cfg['target_size']
    relu_threshold = cfg['relu_threshold']
    if cfg['data_name'] in ['CIFAR10', 'CIFAR100']:
        model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=target_size, relu_threshold=relu_threshold)
    elif cfg['data_name'] in ['FEMNIST', 'MNIST']:
        model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=target_size, relu_threshold=relu_threshold)
    else:
        raise ValueError('wrong dataset name')
    model.apply(init_param)
    return model


