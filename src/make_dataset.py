import os
import torch
from torchvision import transforms
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset
from model import make_model
from module import save, makedir_exist_ok, process_control

if __name__ == "__main__":
    data_names = ['fpb-sa', 'wikisql', 'samsum', 'e2enlg', 'webnlg-2017', 'dart', 'glue-cola', 'glue-mnli', 'glue-mrpc',
                  'glue-qnli', 'glue-qqp', 'glue-rte', 'glue-sst2', 'glue-stsb', 'dolly-15k', 'MNIST', 'CIFAR10',
                  'dreambooth-dog']
    cfg['seed'] = 0
    with torch.no_grad():
        for data_name in data_names:
            cfg['control']['data_name'] = data_name
            if data_name in ['MNIST', 'CIFAR10']:
                cfg['control']['task_name'] = 'ic'
                cfg['control']['model_name'] = 'linear'
            process_control()
            dataset = make_dataset(cfg['data_name'],
                                   cfg['subset_name'])
            model, tokenizer = make_model(cfg['model_name'])
            dataset = process_dataset(dataset, tokenizer)
            print('{}: {}'.format(data_name, cfg['data_size']))
