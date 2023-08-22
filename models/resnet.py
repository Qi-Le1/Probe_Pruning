import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn, CustomReLU
from config import cfg






class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(Block, self).__init__()
        # Because the Batch Normalization is done over the C dimension, computing statistics on (N, H, W) slices
        # C from an expected input of size (N, C, H, W)
        # self.n1 = nn.BatchNorm2d(in_planes)
        if cfg['norm'] == 'bn':
            self.n1 = nn.BatchNorm2d(in_planes)
        elif cfg['norm'] == 'ln':
            self.n1 = nn.GroupNorm(1, in_planes)
        else:
            raise ValueError('wrong norm')
        # print(f'in_planes: {in_planes}')
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        if cfg['norm'] == 'bn':
            self.n2 = nn.BatchNorm2d(planes)
        elif cfg['norm'] == 'ln':
            self.n2 = nn.GroupNorm(1, planes)
        else:
            raise ValueError('wrong norm')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

        self.relu1 = CustomReLU()
        self.relu2 = CustomReLU()

    def forward(self, x):
        # out = F.relu(self.n1(x))
        out, _ = self.relu1(self.n1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # out = self.conv2(F.relu(self.n2(out)))
        out, _ = self.relu2(self.n2(out))
        out = self.conv2(out)
        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, target_size):
        super().__init__()

        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2)
        if cfg['norm'] == 'bn':
            self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion)
        elif cfg['norm'] == 'ln':
            self.n4 = nn.GroupNorm(1, hidden_size[3] * block.expansion)
        else:
            raise ValueError('wrong norm')
        self.relu3 = CustomReLU()
        self.linear = nn.Linear(hidden_size[3] * block.expansion, target_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        # [1, 1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def f(self, x, start_layer_idx):
        if start_layer_idx == -1:
            return self.linear(x)
        
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = F.relu(self.n4(x))
        x, _ = self.relu3(self.n4(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

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


def resnet9():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['resnet9']['hidden_size']
    model = ResNet(data_shape, hidden_size, Block, [1, 1, 1, 1], target_size)
    model.apply(init_param)
    return model


def resnet18():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['resnet18']['hidden_size']
    model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], target_size)
    model.apply(init_param)
    return model
