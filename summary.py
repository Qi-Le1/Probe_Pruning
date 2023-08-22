import argparse
from collections import OrderedDict
import models
import os
from config import cfg, process_args
from tabulate import tabulate
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from data import fetch_dataset, make_data_loader
from utils.api import save, makedir_exist_ok, to_device, process_dataset, collate
from utils.process_command import process_command

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

is_generator = False

def main():
    process_command()
    cfg['seed'] = 0
    runExperiment()
    return

def create_model(track_running_stats=False, on_cpu=False):
    from config import cfg
    if cfg['model_name'] == 'resnet9':
        model = resnet9()
    elif cfg['model_name'] == 'resnet18':
        model = resnet18()
    elif cfg['model_name'] == 'cnn':
        model = create_CNN()
    else:
        raise ValueError('model_name is wrong')
    
    model.to(cfg["device"])

    if on_cpu:
        model.to('cpu')
    # model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=track_running_stats))
    return model

def create_generative_model(dataset_name, embedding=False):
    from config import cfg
    # passed_dataset=dataset_name
    # assert any([alg in algorithm for alg in ['fedgen', 'fedgen']])
    # if 'FedGen' in algorithm:
    #     # temporary roundabout to figure out the sensitivity of the generator network & sampling size
    #     if 'cnn' in algorithm:
    #         gen_model = algorithm.split('-')[1]
    #         passed_dataset+='-' + gen_model
    #     elif '-gen' in algorithm: # we use more lightweight network for sensitivity analysis
    #         passed_dataset += '-cnn1'
    model = Generator(dataset_name, embedding=embedding, latent_layer_idx=-1)
    model.to(cfg["device"])
    return model

def runExperiment():
    cfg['data_name'] = 'FEMNIST'
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    cfg['summary'] = {}
    cfg['summary']['batch_size'] = {'train': 2, 'test': 2}
    cfg['summary']['shuffle'] = {'train': False, 'test': False}
    data_loader = make_data_loader(dataset, 'summary')
    # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))


    



    # model = create_generative_model('CIFAR100')
    # cfg['model_name'] = 'cnn'
    cfg['model_name'] = 'resnet18'
    model = create_model()
    summary = summarize(data_loader['train'], model)
    content, total = parse_summary(summary)
    print(content)
    save_result = total
    save_tag = '{}_{}_{}'.format(cfg['data_name'], cfg['model_name'], cfg['control_name'])
    save(save_result, './output/result/{}.pt'.format(save_tag))
    return


def make_size(input, output):
    if isinstance(input, (tuple, list)):
        return make_size(input[0], output)
    if isinstance(output, (tuple, list)):
        return make_size(input, output[0])
    input_size, output_size = list(input.size()), list(output.size())
    return input_size, output_size


def make_flops(module, input, output):
    if isinstance(input, tuple):
        return make_flops(module, input[0], output)
    if isinstance(output, tuple):
        return make_flops(module, input, output[0])
    flops = compute_flops(module, input, output)
    return flops


def summarize(data_loader, model):
    def register_hook(module):

        def hook(module, input, output):
            module_name = str(module.__class__.__name__)
            if module_name not in summary['count']:
                summary['count'][module_name] = 1
            else:
                summary['count'][module_name] += 1
            key = str(hash(module))
            if key not in summary['module']:
                summary['module'][key] = OrderedDict()
                summary['module'][key]['module_name'] = '{}_{}'.format(module_name, summary['count'][module_name])
                summary['module'][key]['input_size'] = []
                summary['module'][key]['output_size'] = []
                summary['module'][key]['params'] = {}
                summary['module'][key]['flops'] = make_flops(module, input, output)
            input_size, output_size = make_size(input, output)
            summary['module'][key]['input_size'].append(input_size)
            summary['module'][key]['output_size'].append(output_size)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    if name in ['weight']:
                        if name not in summary['module'][key]['params']:
                            summary['module'][key]['params']['weight'] = {}
                            summary['module'][key]['params']['weight']['size'] = list(param.size())
                            summary['module'][key]['coordinates'] = []
                            summary['module'][key]['params']['weight']['mask'] = torch.zeros(
                                summary['module'][key]['params']['weight']['size'], dtype=torch.long)
                    elif name in ['bias']:
                        if name not in summary['module'][key]['params']:
                            summary['module'][key]['params']['bias'] = {}
                            summary['module'][key]['params']['bias']['size'] = list(param.size())
                            summary['module'][key]['params']['bias']['mask'] = torch.zeros(
                                summary['module'][key]['params']['bias']['size'], dtype=torch.long)
                    else:
                        continue
            if len(summary['module'][key]['params']) == 0:
                return
            for name in summary['module'][key]['params']:
                summary['module'][key]['params'][name]['mask'] += 1
            return

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) \
                and not isinstance(module, nn.ModuleDict) and module != model:
            hooks.append(module.register_forward_hook(hook))
        return

    run_mode = True
    summary = OrderedDict()
    summary['module'] = OrderedDict()
    summary['count'] = OrderedDict()
    hooks = []
    model.train(run_mode)
    model.apply(register_hook)
    for i, input in enumerate(data_loader):
        input = collate(input)
        input = to_device(input, cfg['device'])
        if is_generator:
            input = torch.as_tensor([item for item in range(10)])
            input = to_device(input, cfg['device'])
        model(input)
        break
    for h in hooks:
        h.remove()
    summary['total_num_params'] = 0
    summary['total_num_flops'] = 0
    for key in summary['module']:
        num_params = 0
        num_flops = 0
        for name in summary['module'][key]['params']:
            num_params += (summary['module'][key]['params'][name]['mask'] > 0).sum().item()
            num_flops += summary['module'][key]['flops']
        summary['total_num_params'] += num_params
        summary['total_num_flops'] += num_flops
    summary['total_space'] = summary['total_num_params'] * 32. / 8 / (1024 ** 2.)
    return summary


def divide_by_unit(value):
    if value > 1e9:
        return '{:.6} G'.format(value / 1e9)
    elif value > 1e6:
        return '{:.6} M'.format(value / 1e6)
    elif value > 1e3:
        return '{:.6} K'.format(value / 1e3)
    return '{:.6}'.format(value / 1.0)


def parse_summary(summary):
    content = ''
    headers = ['Module Name', 'Input Size', 'Weight Size', 'Output Size', 'Parameters', 'FLOPs']
    records = []
    for key in summary['module']:
        if not summary['module'][key]['params']:
            continue
        module_name = summary['module'][key]['module_name']
        input_size = str(summary['module'][key]['input_size'])
        weight_size = str(summary['module'][key]['params']['weight']['size']) if (
                'weight' in summary['module'][key]['params']) else 'N/A'
        output_size = str(summary['module'][key]['output_size'])
        num_params = 0
        for name in summary['module'][key]['params']:
            num_params += (summary['module'][key]['params'][name]['mask'] > 0).sum().item()
        num_flops = divide_by_unit(summary['module'][key]['flops'])
        records.append([module_name, input_size, weight_size, output_size, num_params, num_flops])
    total_num_param = '{} ({})'.format(summary['total_num_params'], divide_by_unit(summary['total_num_params']))
    total_num_flops = '{} ({})'.format(summary['total_num_flops'], divide_by_unit(summary['total_num_flops']))
    total_space = summary['total_space']
    total = {'num_params': summary['total_num_params'], 'num_flops': summary['total_num_flops'],
             'space': summary['total_space']}
    table = tabulate(records, headers=headers, tablefmt='github')
    content += table + '\n'
    content += '================================================================\n'
    content += 'Total Number of Parameters: {}\n'.format(total_num_param)
    content += 'Total Number of FLOPs: {}\n'.format(total_num_flops)
    content += 'Total Space (MB): {:.2f}\n'.format(total_space)
    makedir_exist_ok('./output')
    content_file = open('./output/summary.md', 'w')
    content_file.write(content)
    content_file.close()
    return content, total


def compute_flops(module, inp, out):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_flops(module, inp, out)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
        return compute_Norm_flops(module, inp, out)
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_flops(module, inp, out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.GELU)):
        return compute_ReLU_flops(module, inp, out)
    elif isinstance(module, nn.Upsample):
        return compute_Upsample_flops(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_flops(module, inp, out)
    else:
        print(f"[Flops]: {type(module).__name__} is not supported!")
        return 0
    pass


def compute_Conv2d_flops(module, inp, out):
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups
    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w
    total_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count
    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_Norm_flops(module, inp, out):
    assert isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm))
    norm_flops = np.prod(inp.shape).item()
    if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)) and module.affine:
        norm_flops *= 2
    if isinstance(module, nn.LayerNorm) and module.elementwise_affine:
        norm_flops *= 2
    return norm_flops


def compute_ReLU_flops(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.GELU))
    batch_size = inp.size()[0]
    active_elements_count = batch_size
    for s in inp.size()[1:]:
        active_elements_count *= s
    return active_elements_count


def compute_Pool2d_flops(module, inp, out):
    assert isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    return np.prod(inp.shape).item()


def compute_Linear_flops(module, inp, out):
    assert isinstance(module, nn.Linear)
    batch_size = np.prod(inp.size()[:-1]).item()
    return batch_size * inp.size()[-1] * out.size()[-1]


def compute_Upsample_flops(module, inp, out):
    assert isinstance(module, nn.Upsample)
    output_size = out[0]
    batch_size = inp.size()[0]
    output_elements_count = batch_size
    for s in output_size.shape[1:]:
        output_elements_count *= s
    return output_elements_count









class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=62):
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
    
    def f(self, x, start_layer_idx):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.n1(x)
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.n2(x)
        # x = x.view(-1, 16 * 4 * 4)
        if start_layer_idx == -1:
            return self.fc3(x)
        
        x = self.n1(self.conv1(x))
        x = self.pool(F.relu(x))
        # print('normalize', flush=True)
        x = self.n2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
        # output['loss'] = loss_fn(output['target'], input['target'])
        return output

class SimpleCNN(nn.Module):
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
        # output['loss'] = loss_fn(output['target'], input['target'])
        return output

def create_CNN():
    model = None
    target_size = cfg['target_size']
    if cfg['data_name'] in ['CIFAR10', 'CIFAR100']:
        model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=target_size)
    elif cfg['data_name'] in ['FEMNIST', 'MNIST']:
        model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=target_size)
    else:
        raise ValueError('wrong dataset name')
    # model.apply(init_param)
    return model



class Generator(nn.Module):
    def __init__(self, dataset_name, embedding=False, latent_layer_idx=-1):
        super(Generator, self).__init__()
        print("Dataset {}".format(dataset_name))
        dataset_name = f"{dataset_name}_{cfg['model_name']}"
        self.embedding = embedding
        # self.dataset = dataset
        #self.model=model
        self.latent_layer_idx = latent_layer_idx
        # TODO: 要修改
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = GENERATORCONFIGS[dataset_name]
        input_dim = self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def init_loss_fn(self):
        self.crossentropy_loss=nn.NLLLoss(reduce=False) # same as above
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()

    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        ### FC modules ####
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        ### Representation layer
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels, latent_layer_idx=-1, verbose=True):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        result = {}
        batch_size = labels.shape[0]
        # print("Batch size {}".format(batch_size))
        # print("Labels {}".format(labels))
        labels = to_device(labels, cfg['device'])
        eps = to_device(torch.rand((batch_size, self.noise_dim)), cfg['device']) # sampling from Gaussian
        if verbose:
            result['eps'] = eps
        if self.embedding: # embedded dense vector
            y_input = self.embedding_layer(labels)
        else: # one-hot (sparse) vector
            y_input = to_device(torch.FloatTensor(batch_size, self.n_class), cfg['device'])
            # print("Y input {}".format(y_input.shape))
            y_input.zero_()
            #labels = labels.view
            y_input.scatter_(1, labels.view(-1,1), 1)

            # print("Y input {}".format(y_input))
        z = to_device(torch.cat((eps, y_input), dim=1), cfg['device'])
        # print(f'zzzz: {z.is_cuda}')
        ### FC layers
        # print(f'self.fc_layers: {self.fc_layers}')
        for layer in self.fc_layers:
            z = layer(z)
            # print(f'zzzz2: {z.is_cuda}')
        z = self.representation_layer(z)
        result['output'] = z
        return result

    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std
#
# class Decoder(nn.Module):
#     """
#     Decoder for both unstructured and image datasets.
#     """
#     def __init__(self, dataset='mnist', latent_layer_idx=-1, n_layers=2, units=32):
#         """
#         Class initializer.
#         """
#         #in_features, out_targets, n_layers=2, units=32):
#         super(Decoder, self).__init__()
#         self.cv_configs, self.input_channel, self.n_class, self.scale, self.noise_dim = GENERATORCONFIGS[dataset]
#         self.hidden_dim = self.scale * self.scale * self.cv_configs[0]
#         self.latent_dim = self.cv_configs[0] * 2
#         self.represent_dims = [self.hidden_dim, self.latent_dim]
#         in_features = self.represent_dims[latent_layer_idx]
#         out_targets = self.noise_dim
#
#         # build layer structure
#         layers = [nn.Linear(in_features, units),
#                   nn.ELU(),
#                   nn.BatchNorm1d(units)]
#
#         for _ in range(n_layers):
#             layers.extend([
#                 nn.Linear(units, units),
#                 nn.ELU(),
#                 nn.BatchNorm1d(units)])
#
#         layers.append(nn.Linear(units, out_targets))
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         """
#         Forward propagation.
#         """
#         out = x.view((x.size(0), -1))
#         out = self.layers(out)
#         return out

class DivLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward2(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        chunk_size = layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer=layer.view((layer.size(0), -1))
        chunk_size=layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

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
        # self.n2 = nn.BatchNorm2d(planes)
        if cfg['norm'] == 'bn':
            self.n2 = nn.BatchNorm2d(planes)
        elif cfg['norm'] == 'ln':
            self.n2 = nn.GroupNorm(1, planes)
        else:
            raise ValueError('wrong norm')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(x))
        # print(f'out1: {out}')
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # print(f'shortcut: {shortcut}')
        out = self.conv1(out)
        # print(f'out2: {out}')
        out = self.conv2(F.relu(self.n2(out)))
        # print(f'out3: {out}')
        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, target_size):
        super().__init__()
        # model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], target_size)
        # cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2)
        # self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion)
        if cfg['norm'] == 'bn':
            self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion)
        elif cfg['norm'] == 'ln':
            self.n4 = nn.GroupNorm(1, hidden_size[3] * block.expansion)
        else:
            raise ValueError('wrong norm')
        # print(f'latent_size: {hidden_size[3] * block.expansion}')
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
        # print(f'x1: {x} {x.size()}\n')
        x = self.layer1(x)
        # print(f'x2: {x} \n')
        x = self.layer2(x)
        # print(f'x3: {x} \n')
        x = self.layer3(x)
        # print(f'x4: {x} \n')
        x = self.layer4(x)
        # print(f'x5: {x} \n')
        x = F.relu(self.n4(x))
        # print(f'x6: {x} \n')
        # print(f'x6_dtype: {x.dtype}')
        x = F.adaptive_avg_pool2d(x, 1)
        # print(f'x7: {x} \n')
        x = x.view(x.size(0), -1)
        # print(f'x8: {x} \n')
        x = self.linear(x)
            # print(f'x9: {x} \n')
        # the x is latent vector, and we      
        # calculate predict layer(latent vector)

        return x

    def forward(self, input, start_layer_idx=None):
        if start_layer_idx == -1:
            output = {}
            output['target'] = self.f(input['data'], start_layer_idx)
            return output
        
        output = {}
        output['target'] = self.f(input['data'], start_layer_idx)
        # output['loss'] = loss_fn(output['target'], input['target'])
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
    # model.apply(init_param)
    return model


CONFIGS_ = {
    # input_channel, n_class, hidden_dim, latent_dim
    'cifar': ([16, 'MaxPooling', 32, 'MaxPooling', 'Flatten'], 3, 10, 2048, 64),
    'cifar100-c25': ([32, 'MaxPooling', 64, 'MaxPooling', 128, 'Flatten'], 3, 25, 128, 128),
    'cifar100-c30': ([32, 'MaxPooling', 64, 'MaxPooling', 128, 'Flatten'], 3, 30, 2048, 128),
    'cifar100-c50': ([32, 'MaxPooling', 64, 'MaxPooling', 128, 'Flatten'], 3, 50, 2048, 128),

    'emnist': ([6, 16, 'Flatten'], 1, 26, 784, 32),
    'mnist': ([6, 16, 'Flatten'], 1, 10, 784, 32),
    'mnist_cnn1': ([6, 'MaxPooling', 16, 'MaxPooling', 'Flatten'], 1, 10, 64, 32),
    'mnist_cnn2': ([16, 'MaxPooling', 32, 'MaxPooling', 'Flatten'], 1, 10, 128, 32),
    'celeb': ([16, 'MaxPooling', 32, 'MaxPooling', 64, 'MaxPooling', 'Flatten'], 3, 2, 64, 32),
    'gen_inference_size': 128
}

# temporary roundabout to evaluate sensitivity of the generator
GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    # 'cifar': (512, 32, 3, 10, 64),
    'CIFAR10_cnn': (512, 84, 3, 10, 64),
    'CIFAR100_cnn': (512, 84, 3, 100, 64),
    'FEMNIST_cnn': (256, 84, 1, 50, 32),
    'MNIST_cnn': (256, 84, 1, 10, 32),

    'CIFAR10_resnet18': (1024, 512, 3, 10, 64),
    'CIFAR100_resnet18': (1024, 512, 3, 100, 64),
    'FEMNIST_resnet18': (1024, 512, 1, 50, 32),
    'MNIST_resnet18': (1024, 512, 1, 10, 32),

    # 'cifar': (512, 32, 3, 10, 64),
    'celeb': (128, 32, 3, 2, 32),
    'mnist': (256, 32, 1, 10, 32),
    'mnist-cnn0': (256, 32, 1, 10, 64),
    'mnist-cnn1': (128, 32, 1, 10, 32),
    'mnist-cnn2': (64, 32, 1, 10, 32),
    'mnist-cnn3': (64, 32, 1, 10, 16),
    'emnist': (256, 32, 1, 26, 32),
    'emnist-cnn0': (256, 32, 1, 26, 64),
    'emnist-cnn1': (128, 32, 1, 26, 32),
    'emnist-cnn2': (128, 32, 1, 26, 16),
    'emnist-cnn3': (64, 32, 1, 26, 32),
}



RUNCONFIGS = {
    'MNIST':
        {
            'ensemble_lr': 1e-4,
            # 'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0, # adversarial student loss
            'unique_labels': 10,
            'generative_alpha':10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },
    'FEMNIST':
        {
            'ensemble_lr': 1e-4,
            # 'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0, # adversarial student loss
            'unique_labels': 50,
            'generative_alpha':10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },
    'CIFAR10':
        {
            'ensemble_lr': 3e-4,
            # 'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'unique_labels': 10,    # available labels
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10, # used to regulate user training
            'weight_decay': 1e-2
        },
    'CIFAR100':
        {
            'ensemble_lr': 3e-4,
            # 'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 100,
            'generative_alpha': 10,
            'generative_beta': 10, 
            'weight_decay': 1e-2
        },

    'emnist':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0, # adversarial student loss
            'unique_labels': 26,
            'generative_alpha':10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },

    'mnist':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'unique_labels': 10,    # available labels
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10, # used to regulate user training
            'weight_decay': 1e-2
        },

    'celeb':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 2,
            'generative_alpha': 10,
            'generative_beta': 10, 
            'weight_decay': 1e-2
        },

}

if __name__ == "__main__":
    main()