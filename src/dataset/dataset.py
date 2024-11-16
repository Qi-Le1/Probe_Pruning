import copy
import dataset
import numpy as np
import os
import re
import time
import random
from collections.abc import Mapping
import copy
import torch
from functools import partial
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets, load_from_disk
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import default_data_collator
from torch.nn.utils.rnn import pad_sequence
from config import cfg
from model import make_model
from module import to_device
from datetime import datetime


data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}


def make_dataset(data_name, subset_name=None, verbose=True):
    dataset_ = {}
    if verbose:
        current_time = datetime.now()
        current_time.strftime('%Y-%m-%d %H:%M:%S')
        print('current_path', os.getcwd())
        print('current_time', current_time)
        print('fetching data {}...'.format(data_name))
    if subset_name != 'none' and subset_name is not None:
        root = os.path.join('data', f'{data_name}_{subset_name}')
    else:
        root = os.path.join('data', data_name)
    print('cache_dir', root)
    if data_name in ['MNIST', 'FashionMNIST']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['c4']:
        # please follow the instruction here: https://huggingface.co/datasets/allenai/c4
        dataset_['train'] = load_dataset('json', data_files={'train': 'data/c4/c4-train.00000-of-01024.json.gz'}, split='train[:10%]')
        dataset_['test'] = load_dataset('json', data_files={'train': 'data/c4/c4-train.00000-of-01024.json.gz'}, split='train[:10%]')
    # piqa: piqa
    # siqa: siqa , 
    # arc-e: arc-easy 
    # arc-c: arc-challenge (Clark et al., 2018), 
    # hellaswag: hellaswag (Zellers et al., 2019) 
    # winogrande: winogrande 
    # obqa: OpenBookQA (Mihaylov et al., 2018)
    elif data_name in ['wikitext', 'arc', 'obqa']:   
        dataset_['test'] = load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], split='test', cache_dir=root)
    elif data_name in ['wikivalid'] and cfg['calibration_stage'] == True:
        dataset_['train'] = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation', cache_dir=root)
    elif data_name in ['wikitest'] and cfg['calibration_stage'] == True:
        dataset_['train'] = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=root)
    elif data_name in ['piqa', 'siqa', 'hellaswag', 'winogrande', 'boolq', 'ptb']:
        dataset_['test'] = load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], split='validation')
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    if "70b" in cfg['model_name'] or "65b" in cfg['model_name']:
        limit = 2000
        if 'train' in dataset_:
            dataset_['train'] = dataset_['train'].select(range(limit))
        if 'validation' in dataset_:
            dataset_['validation'] = dataset_['validation'].select(range(limit))
        if 'test' in dataset_:
            dataset_['test'] = dataset_['test'].select(range(limit))
    return dataset_


def input_collate(batch):
    return {key: [b[key] for b in batch] for key in batch[0]}


def dreambooth_input_collate(batch):
    input_ids = [b["instance_prompt_ids"] for b in batch]
    pixel_values = [b["instance_images"] for b in batch]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if cfg[cfg['model_name']]['prior_loss_weight'] > 0:
        input_ids += [b["class_prompt_ids"] for b in batch]
        pixel_values += [b["class_images"] for b in batch]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch

def torch_default_data_collator(features):
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        

    return batch

def make_data_collate(collate_mode, tokenizer=None):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    elif collate_mode == 'transformer':
        return default_data_collator
        # return torch_default_data_collator
    elif collate_mode == 'dreambooth':
        return dreambooth_input_collate
    elif collate_mode == 'pad':
        return partial(pad_collate, tokenizer=tokenizer)
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, tokenizer, tag, batch_size=None, shuffle=None, sampler=None):
    data_loader = {}
    if 'num_steps' not in cfg:
        cfg['num_steps'] = {}
    if 'dataset_size' not in cfg:
        cfg['dataset_size'] = {}
    for k in dataset:
        batch_size_ = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        shuffle_ = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size_, shuffle=shuffle_,
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=make_data_collate(cfg['collate_mode'], tokenizer),
                                        worker_init_fn=np.random.seed(cfg['seed']), drop_last=True)
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size_, sampler=sampler[k],
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=make_data_collate(cfg['collate_mode'], tokenizer),
                                        worker_init_fn=np.random.seed(cfg['seed']), drop_last=True)
        cfg['num_steps'][k] = len(data_loader[k])
        cfg['dataset_size'][k] = len(dataset[k])
        print(f"{tag}_dataset_size", k, cfg['dataset_size'][k])
        print(f"{tag}_batch_num", k, cfg['num_steps'][k])
    return data_loader

def make_calibration_dataloader(tokenizer):
    if cfg['calibration_dataset'] == 'wikivalid':
        dataset = make_dataset('wikivalid')
        dataset = process_calibration_dataset(dataset, tokenizer, 'wiki')
    elif cfg['calibration_dataset'] == 'wikitest':
        dataset = make_dataset('wikitest')
        dataset = process_calibration_dataset(dataset, tokenizer, 'wiki')
    elif cfg['calibration_dataset'] == 'c4':
        dataset = make_dataset('c4')
        dataset = process_calibration_dataset(dataset, tokenizer, 'c4')
    else:
        raise ValueError('Not valid calibration dataset name')

    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    return data_loader

def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input




def process_calibration_dataset(dataset, tokenizer, dataset_name):
    if cfg['task_name'] in ['clm', 'csr', 'mix']:
        processed_calibrate_sample_num = 0
        if dataset_name == 'c4':
            max_length = cfg[cfg['model_name']]['max_length']
            print('max_length', max_length)

            def preprocess_function(examples):   
                nonlocal processed_calibrate_sample_num
                model_inputs = {
                    'input_ids': [],
                    'attention_mask': [],
                    'labels': []
                }
                inputs = examples['text']
                if processed_calibrate_sample_num >= cfg['calibration_nsamples']:
                    return model_inputs
                for _ in range(cfg['calibration_nsamples']):
                    while True:
                        # i = random.randint(0, len(inputs) - 1)
                        i = torch.randint(0, len(inputs), (1,)).item()
                        # i = 1
                        trainenc = tokenizer(inputs[i], return_tensors='pt')
                        # print('trainenc.input_ids.shape[1]', trainenc.input_ids.shape[1])
                        if trainenc.input_ids.shape[1] > max_length:
                            break
                    processed_calibrate_sample_num += 1
                    if processed_calibrate_sample_num > cfg['calibration_nsamples']:
                        return model_inputs
                    i = torch.randint(0, trainenc.input_ids.shape[1] - max_length, (1,)).item()
                    j = i + max_length
                    inp = trainenc.input_ids[0][i:j]
                    tar = inp.clone()
                    tar[:-1] = -100
                    model_inputs['input_ids'].append(inp)
                    model_inputs['attention_mask'].append(trainenc.attention_mask[0][i:j])
                    # print('trainenc.attention_mask[0][i:j]shapoe', trainenc.attention_mask[0][i:j].shape)
                    model_inputs['labels'].append(tar)
                return model_inputs

            processed_dataset = {}
            processed_dataset['train'] = dataset['train'].map(
                preprocess_function,
                batched=True,
                batch_size=356317,
                num_proc=1,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on sampled dataset",
                keep_in_memory=True,
            )
        elif 'wiki' in dataset_name:
            max_length = cfg[cfg['model_name']]['max_length']
            print('max_length', max_length)
            def preprocess_function_test(examples):   
                nonlocal processed_calibrate_sample_num
                all_text = "\n\n".join(examples['text'])

                model_inputs = tokenizer(all_text, return_tensors='pt', truncation=False, padding=False)
                input_ids = model_inputs['input_ids'][0]
                attention_mask = model_inputs['attention_mask'][0]

                num_samples = len(input_ids) // max_length
                input_chunks = []
                mask_chunks = []
                for i in range(cfg['calibration_nsamples']):
                    # i = random.randint(0, len(input_ids) - max_length - 1)
                    i = torch.randint(0, len(input_ids) - max_length, (1,)).item()
                    j = i + max_length
                    input_chunks.append(input_ids[i: j])
                    mask_chunks.append(attention_mask[i: j])
                    processed_calibrate_sample_num += 1
                    if processed_calibrate_sample_num >= cfg['calibration_nsamples']:
                        break
                # input_chunks = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length)]
                # mask_chunks = [attention_mask[i:i + max_length] for i in range(0, len(attention_mask), max_length)]
                final_inputs = {
                    'input_ids': [],
                    'attention_mask': [],
                    'labels': []
                }
                if processed_calibrate_sample_num > cfg['calibration_nsamples'] + 4:
                    return final_inputs
                for i in range(len(input_chunks)):
                    # print('len(input_chunks[i])', len(input_chunks[i]))
                    if len(input_chunks[i]) == max_length:
                        final_inputs['input_ids'].append(input_chunks[i])
                        final_inputs['attention_mask'].append(mask_chunks[i])
                        labels = copy.deepcopy(input_chunks[i])
                        final_inputs['labels'].append(labels)
                
                return final_inputs

            processed_dataset = {}
            processed_dataset['train'] = dataset['train'].map(
                preprocess_function_test,
                batched=True,
                batch_size=100000,
                num_proc=1,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                keep_in_memory=True,
            )
    else:
        raise ValueError('Not valid task name')
    return processed_dataset

def custom_padding(padding_length, input_ids, attention_mask, labels, tokenizer):
    # if 'rightpad' in cfg['prune_method'] or 'dense' in cfg['prune_method']:
    input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
    attention_mask = attention_mask + [0] * padding_length
    labels = labels + [-100] * padding_length
    # else:
    #     input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
    #     attention_mask = [0] * padding_length + attention_mask
    #     labels = [-100] * padding_length + labels
    return input_ids, attention_mask, labels

# last 2 arguments only for mixing task (clm and csr in 1 batch)
def process_dataset(dataset, tokenizer, data_name=None, clm_num_samples=None, csr_data_loader=None):
    processed_c4_sample_num = 0
    if cfg['task_name'] in ['clm', 'csr', 'mix']:
        text_column = cfg['text_column']
        label_column = cfg['label_column']
        if cfg['data_name'] == 'c4':
            max_length = cfg[cfg['model_name']]['max_length']
            # enter here only for dense model c4 flops info
            # fix 2000 samples for gridsearch
            cfg['calibration_nsamples'] = 2000
            print('max_length', max_length)

            def preprocess_function(examples):   
                nonlocal processed_c4_sample_num
                model_inputs = {
                    'input_ids': [],
                    'attention_mask': [],
                    'labels': []
                }
                inputs = examples['text']
                if processed_c4_sample_num >= cfg['calibration_nsamples']:
                    return model_inputs
                for _ in range(cfg['calibration_nsamples']):
                    while True:
                        # i = random.randint(0, len(inputs) - 1)
                        i = torch.randint(0, len(inputs), (1,)).item()
                        # i = 1
                        trainenc = tokenizer(inputs[i], return_tensors='pt')
                        # print('trainenc.input_ids.shape[1]', trainenc.input_ids.shape[1])
                        if trainenc.input_ids.shape[1] > max_length:
                            break
                    processed_c4_sample_num += 1
                    if processed_c4_sample_num > cfg['calibration_nsamples']:
                        return model_inputs
                    i = torch.randint(0, trainenc.input_ids.shape[1] - max_length, (1,)).item()
                    j = i + max_length
                    inp = trainenc.input_ids[0][i:j]
                    tar = inp.clone()
                    tar[:-1] = -100
                    model_inputs['input_ids'].append(inp)
                    model_inputs['attention_mask'].append(trainenc.attention_mask[0][i:j])
                    # print('trainenc.attention_mask[0][i:j]shapoe', trainenc.attention_mask[0][i:j].shape)
                    model_inputs['labels'].append(tar)
                return model_inputs

            processed_dataset = {}
            processed_dataset['test'] = dataset['test'].map(
                preprocess_function,
                batched=True,
                batch_size=356317,
                num_proc=1,
                remove_columns=dataset["test"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on sampled dataset",
                keep_in_memory=True,
            )
            processed_calibrate_sample_num = 0
        elif cfg['data_name'] == 'wikitext' or data_name == 'wikitext':
            max_length = cfg[cfg['model_name']]['max_length']
            print('max_length', max_length)
            def preprocess_function_test(examples):   
                print('times')
                all_text = "\n\n".join(examples[text_column[0]])

                model_inputs = tokenizer(all_text, return_tensors='pt', truncation=False, padding=False)

                input_ids = model_inputs['input_ids'][0]
                attention_mask = model_inputs['attention_mask'][0]

                num_samples = len(input_ids) // max_length
                if 'mix' in cfg['prune_method'] and clm_num_samples is not None:
                    num_samples = clm_num_samples
                input_chunks = []
                mask_chunks = []
                if 'inorderwiki' in cfg['prune_method']:
                    input_chunks = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length)]
                    mask_chunks = [attention_mask[i:i + max_length] for i in range(0, len(attention_mask), max_length)]
                else:
                    for i in range(num_samples):
                        i = torch.randint(0, len(input_ids) - max_length, (1,)).item()
                        j = i + max_length
                        input_chunks.append(input_ids[i: j])
                        mask_chunks.append(attention_mask[i: j])
                
                final_inputs = defaultdict(list)
                if cfg['task_name'] == 'mix' and clm_num_samples is not None:
                    for i, input in enumerate(csr_data_loader['test']):
                        print('i', i)
                        csr_input_ids = input['input_ids'].tolist()
                        print('csr_input_ids', csr_input_ids, len(csr_input_ids))
                        csr_attention_mask = input['attention_mask'].tolist()
                        csr_labels = input['labels'].tolist()
                        csr_input_indices = input['input_indices'].tolist()
                        csr_correct_labels = input['correct_labels'].tolist()
                        
                        final_inputs['input_ids'].extend(csr_input_ids)
                        final_inputs['attention_mask'].extend(csr_attention_mask)
                        final_inputs['labels'].extend(csr_labels)
                        final_inputs['input_indices'].extend(csr_input_indices)
                        final_inputs['correct_labels'].extend(csr_correct_labels)
                        print('lencsr_input_indices', len(csr_input_indices))
                        print('lencorrect_labels', len(csr_correct_labels))
                        if len(csr_input_indices) != int(0.5 * cfg['batch_size']):
                            print('not equal')


                        clm_batch_size = int(0.5 * cfg['batch_size'])
                        start_indx = (i * clm_batch_size) % len(input_chunks)
                        end_indx = ((i + 1) * clm_batch_size) % len(input_chunks)
                        # clm_input_ids = input_chunks[start_indx:end_indx]
                        # print('start_indx', start_indx, len(input_chunks))
                        # print('end_indx', end_indx)
                        # print('clm_input_ids', clm_input_ids, len(clm_input_ids))
                        # clm_attention_mask = mask_chunks[start_indx:end_indx]
                        # clm_labels = copy.deepcopy(clm_input_ids)
                        if end_indx < start_indx:
                            # This means we've wrapped around, so concatenate the end and start of the list
                            clm_input_ids = input_chunks[start_indx:] + input_chunks[:end_indx]
                            clm_attention_mask = mask_chunks[start_indx:] + mask_chunks[:end_indx]
                        else:
                            # Normal slicing
                            clm_input_ids = input_chunks[start_indx:end_indx]
                            clm_attention_mask = mask_chunks[start_indx:end_indx]
                        clm_labels = copy.deepcopy(clm_input_ids)

                        final_inputs['input_ids'].extend(clm_input_ids)
                        final_inputs['attention_mask'].extend(clm_attention_mask)
                        final_inputs['labels'].extend(clm_labels)
                        final_inputs['input_indices'].extend([i] * clm_batch_size)
                        final_inputs['correct_labels'].extend([1] * clm_batch_size)
                        print('------s')
                        print(len(final_inputs['input_ids']))
                        print(len(final_inputs['attention_mask']))
                        print(len(final_inputs['labels']))
                        print(len(final_inputs['input_indices']))
                        print(len(final_inputs['correct_labels']))
                        assert len(final_inputs['input_ids']) == len(final_inputs['attention_mask']) == len(final_inputs['labels']) == len(final_inputs['input_indices']) == len(final_inputs['correct_labels'])
                else:
                    for i in range(len(input_chunks)):
                        # print('len(input_chunks[i])', len(input_chunks[i]))
                        if len(input_chunks[i]) == max_length:
                            final_inputs['input_ids'].append(input_chunks[i])
                            final_inputs['attention_mask'].append(mask_chunks[i])
                            labels = copy.deepcopy(input_chunks[i])
                            final_inputs['labels'].append(labels)
                
                return final_inputs

            processed_dataset = {}
            processed_dataset['test'] = dataset['test'].map(
                preprocess_function_test,
                batched=True,
                batch_size=100000,
                num_proc=1,
                remove_columns=dataset["test"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                keep_in_memory=True,
            )
        elif cfg['data_name'] == 'ptb':
            max_length = cfg[cfg['model_name']]['max_length']
            print('max_length', max_length)
            def preprocess_function(examples):   
                all_text = "\n\n".join(examples['sentence'])

                model_inputs = tokenizer(all_text, return_tensors='pt', truncation=False, padding=False)

                input_ids = model_inputs['input_ids'][0]
                attention_mask = model_inputs['attention_mask'][0]

                num_samples = len(input_ids) // max_length
                input_chunks = []
                mask_chunks = []
                for i in range(num_samples):
                    # i = random.randint(0, len(input_ids) - max_length - 1)
                    i = torch.randint(0, len(input_ids) - max_length, (1,)).item()
                    j = i + max_length
                    input_chunks.append(input_ids[i: j])
                    mask_chunks.append(attention_mask[i: j])
                final_inputs = defaultdict(list)
                for i in range(len(input_chunks)):
                    if len(input_chunks[i]) == max_length:
                        final_inputs['input_ids'].append(input_chunks[i])
                        final_inputs['attention_mask'].append(mask_chunks[i])
                        labels = copy.deepcopy(input_chunks[i])
                        final_inputs['labels'].append(labels)
                
                return final_inputs

            processed_dataset = {}
            processed_dataset['test'] = dataset['test'].map(
                preprocess_function,
                batched=True,
                batch_size=100000,
                num_proc=1,
                remove_columns=dataset["test"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                keep_in_memory=True,
            )
        # boolq
        # piqa: piqa
        # siqa: siqa , 
        # arc-e: arc-easy 
        # arc-c: arc-challenge (Clark et al., 2018), 
        # hellaswag: hellaswag (Zellers et al., 2019) 
        # obqa: OpenBookQA (Mihaylov et al., 2018)
        # for csr, cant use Padtoken in the first place, otherwise the loss will be nan
        # for csr, cant use Padtoken in the first place, otherwise the loss will be nan
        # for csr, cant use Padtoken in the first place, otherwise the loss will be nan
        elif cfg['data_name'] == 'boolq':
            '''
            {
                "answer": false,
                "passage": "\"All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned...",
                "question": "does ethanol take more energy make that produces"
            }
            '''

            def yesno(x):
                if x:
                    return "yes"
                else:
                    return "no"
    
            max_length = cfg[cfg['model_name']]['max_length']
            cur_input_bsz = cfg['batch_size']
            def tokenize_function(examples):
                batch_size = len(examples['answer'])
                targets = examples['answer']

                inputs = []
                labels = []
                correct_labels_extended = []
                input_indices = []
                for i in range(batch_size):
                    cur_correct_label = 1 if yesno(targets[i]) == 'yes' else 0
                    inputs.extend([f"{examples['passage'][i]}\nQuestion: {examples['question'][i]}?\nAnswer:"])
                    labels.extend([' yes'])
                    correct_labels_extended.extend([cur_correct_label])
                    input_indices.extend([i])

                    cur_correct_label = 1 if yesno(targets[i]) == 'no' else 0
                    inputs.extend([f"{examples['passage'][i]}\nQuestion: {examples['question'][i]}?\nAnswer:"])
                    labels.extend([' no'])
                    correct_labels_extended.extend([cur_correct_label])
                    input_indices.extend([i])

                tokenizer.truncation_side = 'left'
                model_inputs = tokenizer(inputs, max_length=max_length, padding="do_not_pad", truncation=True)
                tokenizer.truncation_side = 'right'
                labels = tokenizer(labels, max_length=max_length, padding="do_not_pad", truncation=True)

                for i in range(len(correct_labels_extended)):
                    if labels["input_ids"][i][0] == tokenizer.bos_token_id:
                        labels["input_ids"][i] = labels["input_ids"][i][1:]  # skip the first token
                        labels["attention_mask"][i] = labels["attention_mask"][i][1:]

                for batch_index in range(0, len(correct_labels_extended), cur_input_bsz):
                    # Determine the batch boundaries
                    batch_end = min(batch_index + cur_input_bsz, len(correct_labels_extended))

                    # Initialize a variable to track the maximum length of sequences in this batch
                    max_length_in_batch = 0

                    # First pass: calculate the maximum length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        # Combine the current input ids and label input ids
                        temp_input = sample_input_ids + label_input_ids
                        max_length_in_batch = max(max_length_in_batch, len(temp_input))

                    # Second pass: adjust sequences to the max length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        temp_input = sample_input_ids + label_input_ids
                        print('len(temp_input)', len(temp_input))
                        temp_attention_mask = sample_attention_mask + label_attention_mask
                        temp_label = [-100] * len(sample_input_ids) + label_input_ids

                        # Truncate or pad sequences based on the batch's maximum length
                        if len(temp_input) > max_length_in_batch:
                            temp_input = temp_input[-max_length_in_batch:]
                            temp_attention_mask = temp_attention_mask[-max_length_in_batch:]
                            temp_label = temp_label[-max_length_in_batch:]
                        else:
                            padding_length = max_length_in_batch - len(temp_input)
                            temp_input, temp_attention_mask, temp_label = custom_padding(padding_length, temp_input, temp_attention_mask, temp_label, tokenizer)

                        model_inputs["input_ids"][i] = torch.tensor(temp_input[-max_length:])
                        model_inputs["attention_mask"][i] = torch.tensor(temp_attention_mask[-max_length:])
                        labels["input_ids"][i] = torch.tensor(temp_label[-max_length:])

                model_inputs["labels"] = labels["input_ids"]
                model_inputs['input_indices'] = input_indices
                model_inputs["correct_labels"] = correct_labels_extended
                return model_inputs

            processed_dataset = {}
            processed_dataset['test'] = dataset['test'].map(
                tokenize_function,
                batched=True,
                # batch_size=50,
                batch_size=100000,
                num_proc=1,
                remove_columns=dataset["test"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                keep_in_memory=True,
            )
        elif cfg['data_name'] == 'piqa':
            '''
            {
                "goal": "How do I ready a guinea pig cage for it's new occupants?",
                "sol1": "Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.",
                "sol2": "Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.",
                "label": 0,
            }
            '''
            max_length = cfg[cfg['model_name']]['max_length']
            cur_input_bsz = cfg['batch_size']
            def tokenize_function(examples):
                batch_size = len(examples['label'])
                targets = examples['label']

                inputs = []
                labels = []
                correct_labels_extended = []
                input_indices = []
                for i in range(batch_size):
                    for j in range(2):
                        goal = examples['goal'][i]
                        sol = examples[f"sol{j+1}"][i]
                        inputs.extend([goal])
                        labels.extend([sol])
                        correct_labels_extended.extend([targets[i]])
                        input_indices.extend([i])
                # inputs = [(f"{' '.join([f'{col}: {examples[col][i]}' for col in text_column])}" f'{label_column}: ') for i in
                #           range(batch_size)]
                tokenizer.truncation_side = 'left'
                model_inputs = tokenizer(inputs, max_length=max_length, padding="do_not_pad", truncation=True)
                tokenizer.truncation_side = 'right'
                labels = tokenizer(labels, max_length=max_length, padding="do_not_pad", truncation=True)

                for i in range(len(correct_labels_extended)):
                    if labels["input_ids"][i][0] == tokenizer.bos_token_id:
                        labels["input_ids"][i] = labels["input_ids"][i][1:]  # skip the first token
                        labels["attention_mask"][i] = labels["attention_mask"][i][1:]

                for batch_index in range(0, len(correct_labels_extended), cur_input_bsz):
                    # Determine the batch boundaries
                    batch_end = min(batch_index + cur_input_bsz, len(correct_labels_extended))

                    # Initialize a variable to track the maximum length of sequences in this batch
                    max_length_in_batch = 0

                    # First pass: calculate the maximum length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        # Combine the current input ids and label input ids
                        temp_input = sample_input_ids + label_input_ids
                        max_length_in_batch = max(max_length_in_batch, len(temp_input))

                    # Second pass: adjust sequences to the max length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        temp_input = sample_input_ids + label_input_ids
                        print('len(temp_input)', len(temp_input))
                        temp_attention_mask = sample_attention_mask + label_attention_mask
                        temp_label = [-100] * len(sample_input_ids) + label_input_ids

                        # Truncate or pad sequences based on the batch's maximum length
                        if len(temp_input) > max_length_in_batch:
                            temp_input = temp_input[-max_length_in_batch:]
                            temp_attention_mask = temp_attention_mask[-max_length_in_batch:]
                            temp_label = temp_label[-max_length_in_batch:]
                        else:
                            padding_length = max_length_in_batch - len(temp_input)
                            temp_input, temp_attention_mask, temp_label = custom_padding(padding_length, temp_input, temp_attention_mask, temp_label, tokenizer)


                        model_inputs["input_ids"][i] = torch.tensor(temp_input[-max_length:])
                        model_inputs["attention_mask"][i] = torch.tensor(temp_attention_mask[-max_length:])
                        labels["input_ids"][i] = torch.tensor(temp_label[-max_length:])

                model_inputs["labels"] = labels["input_ids"]
                model_inputs['input_indices'] = input_indices
                model_inputs["correct_labels"] = correct_labels_extended
                return model_inputs

            processed_dataset = {}
            processed_dataset['test'] = dataset['test'].map(
                tokenize_function,
                batched=True,
                # batch_size=50,
                batch_size=100000,
                num_proc=1,
                remove_columns=dataset["test"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                keep_in_memory=True,
            )
        elif cfg['data_name'] == 'siqa':
            '''
            {
                "answerA": "sympathetic",
                "answerB": "like a person who was unable to help",
                "answerC": "incredulous",
                "context": "Sydney walked past a homeless woman asking for change but did not have any money they could give to her. Sydney felt bad afterwards.",
                "label": "1",
                "question": "How would you describe Sydney?"
            }
            '''
            max_length = cfg[cfg['model_name']]['max_length']
            cur_input_bsz = cfg['batch_size']
            def tokenize_function(examples):
                batch_size = len(examples['label'])
                targets = examples['label']
                targets = [int(targets[i])-1 for i in range(batch_size)]

                inputs = []
                labels = []
                correct_labels_extended = []
                input_indices = []
                for i in range(batch_size):
                    for char in ['A', 'B', 'C']:  # Loop through characters A, B, C
                        inputs.extend([examples['question'][i] + ' ' + examples['context'][i]])
                        labels.extend([examples[f'answer{char}'][i]])  # Use the char variable to dynamically refer to answer keys
                        # correct_labels_extended.extend([targets[i]])
                        input_indices.extend([i])
                    correct_label = [0] * 3
                    correct_label[targets[i]] = 1
                    correct_labels_extended.extend(correct_label)

                tokenizer.truncation_side = 'left'
                model_inputs = tokenizer(inputs, max_length=max_length, padding="do_not_pad", truncation=True)
                tokenizer.truncation_side = 'right'
                labels = tokenizer(labels, max_length=max_length, padding="do_not_pad", truncation=True)

                for i in range(len(correct_labels_extended)):
                    if labels["input_ids"][i][0] == tokenizer.bos_token_id:
                        labels["input_ids"][i] = labels["input_ids"][i][1:]  # skip the first token
                        labels["attention_mask"][i] = labels["attention_mask"][i][1:]

                for batch_index in range(0, len(correct_labels_extended), cur_input_bsz):
                    # Determine the batch boundaries
                    batch_end = min(batch_index + cur_input_bsz, len(correct_labels_extended))

                    # Initialize a variable to track the maximum length of sequences in this batch
                    max_length_in_batch = 0

                    # First pass: calculate the maximum length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        # Combine the current input ids and label input ids
                        temp_input = sample_input_ids + label_input_ids
                        max_length_in_batch = max(max_length_in_batch, len(temp_input))

                    # Second pass: adjust sequences to the max length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        temp_input = sample_input_ids + label_input_ids
                        print('len(temp_input)', len(temp_input))
                        temp_attention_mask = sample_attention_mask + label_attention_mask
                        temp_label = [-100] * len(sample_input_ids) + label_input_ids

                        # Truncate or pad sequences based on the batch's maximum length
                        if len(temp_input) > max_length_in_batch:
                            temp_input = temp_input[-max_length_in_batch:]
                            temp_attention_mask = temp_attention_mask[-max_length_in_batch:]
                            temp_label = temp_label[-max_length_in_batch:]
                        else:
                            padding_length = max_length_in_batch - len(temp_input)
                            temp_input, temp_attention_mask, temp_label = custom_padding(padding_length, temp_input, temp_attention_mask, temp_label, tokenizer)

                        model_inputs["input_ids"][i] = torch.tensor(temp_input[-max_length:])
                        model_inputs["attention_mask"][i] = torch.tensor(temp_attention_mask[-max_length:])
                        labels["input_ids"][i] = torch.tensor(temp_label[-max_length:])

                model_inputs["labels"] = labels["input_ids"]
                model_inputs['input_indices'] = input_indices
                model_inputs["correct_labels"] = correct_labels_extended
                return model_inputs

            processed_dataset = {}
            processed_dataset['test'] = dataset['test'].map(
                tokenize_function,
                batched=True,
                # batch_size=50,
                batch_size=100000,
                num_proc=1,
                remove_columns=dataset["test"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                keep_in_memory=True,
            )
        elif cfg['data_name'] == 'arc' or data_name == 'arc':
            '''
            https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/arc.py
            {
                "answerKey": "B",
                "choices": {
                    "label": ["A", "B", "C", "D"],
                    "text": ["Shady areas increased.", "Food sources increased.", "Oxygen levels increased.", "Available water increased."]
                },
                "id": "Mercury_SC_405487",
                "question": "One year, the oak trees in a park began producing more acorns than usual. The next year, the population of chipmunks in the park also increased. Which best explains why there were more chipmunks the next year?"
            }
            '''
            
            max_length = cfg[cfg['model_name']]['max_length']
            cur_input_bsz = cfg['batch_size']
            def tokenize_function(examples):
                batch_size = len(examples['answerKey'])
                targets = examples['answerKey']

                num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
                targets = [num_to_letter.get(target, target) for target in targets]
                # Convert each target to its numerical index
                targets = [["A", "B", "C", "D", "E"].index(target) for target in targets]

                inputs = []
                labels = []
                correct_labels_extended = []
                input_indices = []
                for i in range(batch_size):
                    cur_label = examples['choices'][i]['text']
                    num_choices = len(cur_label)
                    inputs.extend(["Question: " + examples['question'][i] + "\nAnswer:"] * num_choices)
                    labels.extend(cur_label)
                    # correct_labels_extended.extend([targets[i]] * num_choices)
                    correct_label = [0] * num_choices
                    correct_label[targets[i]] = 1
                    correct_labels_extended.extend(correct_label)
                    input_indices.extend([i] * num_choices)
                # inputs = [(f"{' '.join([f'{col}: {examples[col][i]}' for col in text_column])}" f'{label_column}: ') for i in
                #           range(batch_size)]
                # model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length")
                # labels = tokenizer(labels, max_length=max_length, padding="do_not_pad", truncation=True)
                tokenizer.truncation_side = 'left'
                model_inputs = tokenizer(inputs, max_length=max_length, padding="do_not_pad", truncation=True)
                tokenizer.truncation_side = 'right'
                labels = tokenizer(labels, max_length=max_length, padding="do_not_pad", truncation=True)

                for i in range(len(correct_labels_extended)):
                    if labels["input_ids"][i][0] == tokenizer.bos_token_id:
                        labels["input_ids"][i] = labels["input_ids"][i][1:]  # skip the first token
                        labels["attention_mask"][i] = labels["attention_mask"][i][1:]

                for batch_index in range(0, len(correct_labels_extended), cur_input_bsz):
                    # Determine the batch boundaries
                    batch_end = min(batch_index + cur_input_bsz, len(correct_labels_extended))

                    # Initialize a variable to track the maximum length of sequences in this batch
                    max_length_in_batch = 0

                    # First pass: calculate the maximum length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        # Combine the current input ids and label input ids
                        temp_input = sample_input_ids + label_input_ids
                        max_length_in_batch = max(max_length_in_batch, len(temp_input))
                        # align with clm lenth if task is mix
                        if cfg['task_name'] == 'mix':
                            max_length_in_batch = cfg['max_seq_len']
                        # max_length_in_batch = 100

                    # Second pass: adjust sequences to the max length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        temp_input = sample_input_ids + label_input_ids
                        print('len(temp_input)', len(temp_input))
                        temp_attention_mask = sample_attention_mask + label_attention_mask
                        temp_label = [-100] * len(sample_input_ids) + label_input_ids

                        # Truncate or pad sequences based on the batch's maximum length
                        if len(temp_input) > max_length_in_batch:
                            temp_input = temp_input[-max_length_in_batch:]
                            temp_attention_mask = temp_attention_mask[-max_length_in_batch:]
                            temp_label = temp_label[-max_length_in_batch:]
                        else:
                            padding_length = max_length_in_batch - len(temp_input)
                            temp_input, temp_attention_mask, temp_label = custom_padding(padding_length, temp_input, temp_attention_mask, temp_label, tokenizer)


                        # Update the model inputs and labels
                        model_inputs["input_ids"][i] = torch.tensor(temp_input[-max_length:])
                        model_inputs["attention_mask"][i] = torch.tensor(temp_attention_mask[-max_length:])
                        labels["input_ids"][i] = torch.tensor(temp_label[-max_length:])
                    print('max_length_in_batch', max_length_in_batch)


                model_inputs["labels"] = labels["input_ids"]
                model_inputs['input_indices'] = input_indices
                model_inputs["correct_labels"] = correct_labels_extended
                # model_inputs["text_seq"] = input_text
                return model_inputs

            processed_dataset = {}
            processed_dataset['test'] = dataset['test'].map(
                tokenize_function,
                batched=True,
                # batch_size=50,
                batch_size=100000,
                num_proc=1,
                remove_columns=dataset["test"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                keep_in_memory=True,
            )
        elif cfg['data_name'] == 'hellaswag':
            '''
            Reference: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hellaswag.py
            This example was too long and was cropped:
            {
                "activity_label": "Removing ice from car",
                "ctx": "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then",
                "ctx_a": "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles.",
                "ctx_b": "then",
                "endings": "[\", the man adds wax to the windshield and cuts it.\", \", a person board a ski lift, while two men supporting the head of the per...",
                "ind": 4,
                "label": "3",
                "source_id": "activitynet~v_-1IBHYS3L-Y",
                "split": "train",
                "split_type": "indomain"
            }
            '''
            max_length = cfg[cfg['model_name']]['max_length']
            cur_input_bsz = cfg['batch_size']
            def preprocess(text):
                text = text.strip()
                # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
                text = text.replace(" [title]", ". ")
                text = re.sub("\\[.*?\\]", "", text)
                text = text.replace("  ", " ")
                return text
                
            def tokenize_function(examples):
                batch_size = len(examples['label'])
                targets = examples['label']
                examples['ctx_b'] = [examples['ctx_b'][i].capitalize() for i in range(batch_size)]
                targets = [int(targets[i]) for i in range(batch_size)]

                inputs = []
                labels = []
                correct_labels_extended = []
                input_indices = []
                for i in range(batch_size):
                    num_choices = len(examples['endings'][i])
                    inputs.extend([preprocess(examples['activity_label'][i] + ': ' + examples['ctx_a'][i] + " " + examples['ctx_b'][i])] * num_choices)
                    for j in range(num_choices):
                        labels.extend([preprocess(examples['endings'][i][j])])
                    # labels.extend(preprocess(examples['endings'][i]))
                    # correct_labels_extended.extend([targets[i]] * num_choices)
                    correct_label = [0] * num_choices
                    correct_label[targets[i]] = 1
                    correct_labels_extended.extend(correct_label)
                    input_indices.extend([i] * num_choices)
                # inputs = [(f"{' '.join([f'{col}: {examples[col][i]}' for col in text_column])}" f'{label_column}: ') for i in
                #           range(batch_size)]
                tokenizer.truncation_side = 'left'
                model_inputs = tokenizer(inputs, max_length=max_length, padding="do_not_pad", truncation=True)
                tokenizer.truncation_side = 'right'
                labels = tokenizer(labels, max_length=max_length, padding="do_not_pad", truncation=True)

                for i in range(len(correct_labels_extended)):
                    if labels["input_ids"][i][0] == tokenizer.bos_token_id:
                        labels["input_ids"][i] = labels["input_ids"][i][1:]  # skip the first token
                        labels["attention_mask"][i] = labels["attention_mask"][i][1:]

                for batch_index in range(0, len(correct_labels_extended), cur_input_bsz):
                    # Determine the batch boundaries
                    batch_end = min(batch_index + cur_input_bsz, len(correct_labels_extended))

                    # Initialize a variable to track the maximum length of sequences in this batch
                    max_length_in_batch = 0

                    # First pass: calculate the maximum length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        # Combine the current input ids and label input ids
                        temp_input = sample_input_ids + label_input_ids
                        max_length_in_batch = max(max_length_in_batch, len(temp_input))

                    # Second pass: adjust sequences to the max length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        temp_input = sample_input_ids + label_input_ids
                        print('len(temp_input)', len(temp_input))
                        temp_attention_mask = sample_attention_mask + label_attention_mask
                        temp_label = [-100] * len(sample_input_ids) + label_input_ids

                        # Truncate or pad sequences based on the batch's maximum length
                        if len(temp_input) > max_length_in_batch:
                            temp_input = temp_input[-max_length_in_batch:]
                            temp_attention_mask = temp_attention_mask[-max_length_in_batch:]
                            temp_label = temp_label[-max_length_in_batch:]
                        else:
                            padding_length = max_length_in_batch - len(temp_input)
                            temp_input, temp_attention_mask, temp_label = custom_padding(padding_length, temp_input, temp_attention_mask, temp_label, tokenizer)


                        model_inputs["input_ids"][i] = torch.tensor(temp_input[-max_length:])
                        model_inputs["attention_mask"][i] = torch.tensor(temp_attention_mask[-max_length:])
                        labels["input_ids"][i] = torch.tensor(temp_label[-max_length:])

                model_inputs["labels"] = labels["input_ids"]
                model_inputs['input_indices'] = input_indices
                model_inputs["correct_labels"] = correct_labels_extended
                return model_inputs

            processed_dataset = {}
            processed_dataset['test'] = dataset['test'].map(
                tokenize_function,
                batched=True,
                # batch_size=50,
                batch_size=100000,
                num_proc=1,
                remove_columns=dataset["test"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                keep_in_memory=True,
            )
        elif cfg['data_name'] == 'winogrande':
            '''
            Reference: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/winogrande.py
            This example was too long and was cropped:
            {
                "sentence": "John moved the couch from the garage to the backyard to create space. The _ is small.",
                "option1": "garage",
                "option2": "backyard",
                "answer": "1"
            }
            '''
            max_length = cfg[cfg['model_name']]['max_length']
            cur_input_bsz = cfg['batch_size']
            def partial_context(sentence, option):
                # Substitute the pronoun in the sentence with the specified option
                # and ignore everything after.
                pronoun_loc = sentence.index("_")
                return sentence[:pronoun_loc] + option

            def partial_target(sentence):
                # The target is everything after the document specified pronoun.
                pronoun_loc = sentence.index("_") + 1
                return " " + sentence[pronoun_loc:].strip()
    
            answer_to_num = {"1": 0, "2": 1}
            def tokenize_function(examples):
                batch_size = len(examples['answer'])
                targets = examples['answer']
                # examples['ctx_b'] = [examples['ctx_b'][i].capitalize() for i in range(batch_size)]
                targets = [answer_to_num[targets[i]] for i in range(batch_size)]

                inputs = []
                labels = []
                correct_labels_extended = []
                input_indices = []
                for i in range(batch_size):
                    for j in range(2):
                        option = examples[f"option{j+1}"][i]
                        sentence = examples['sentence'][i]
                        # print('sentence', sentence, '||||||', partial_context(sentence, option))
                        inputs.extend([partial_context(sentence, option)])
                        # print('labels',partial_target(sentence) )
                        labels.extend([partial_target(sentence)])
                        # correct_labels_extended.extend([targets[i]])
                        input_indices.extend([i])
                    correct_label = [0] * 2
                    correct_label[targets[i]] = 1
                    correct_labels_extended.extend(correct_label)

                tokenizer.truncation_side = 'left'
                model_inputs = tokenizer(inputs, max_length=max_length, padding="do_not_pad", truncation=True)
                tokenizer.truncation_side = 'right'
                labels = tokenizer(labels, max_length=max_length, padding="do_not_pad", truncation=True)

                for i in range(len(correct_labels_extended)):
                    if labels["input_ids"][i][0] == tokenizer.bos_token_id:
                        labels["input_ids"][i] = labels["input_ids"][i][1:]  # skip the first token
                        labels["attention_mask"][i] = labels["attention_mask"][i][1:]

                for batch_index in range(0, len(correct_labels_extended), cur_input_bsz):
                    # Determine the batch boundaries
                    batch_end = min(batch_index + cur_input_bsz, len(correct_labels_extended))

                    # Initialize a variable to track the maximum length of sequences in this batch
                    max_length_in_batch = 0

                    # First pass: calculate the maximum length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        # Combine the current input ids and label input ids
                        temp_input = sample_input_ids + label_input_ids
                        max_length_in_batch = max(max_length_in_batch, len(temp_input))

                    # Second pass: adjust sequences to the max length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        temp_input = sample_input_ids + label_input_ids
                        print('len(temp_input)', len(temp_input))
                        temp_attention_mask = sample_attention_mask + label_attention_mask
                        temp_label = [-100] * len(sample_input_ids) + label_input_ids

                        # Truncate or pad sequences based on the batch's maximum length
                        if len(temp_input) > max_length_in_batch:
                            temp_input = temp_input[-max_length_in_batch:]
                            temp_attention_mask = temp_attention_mask[-max_length_in_batch:]
                            temp_label = temp_label[-max_length_in_batch:]
                        else:
                            padding_length = max_length_in_batch - len(temp_input)
                            temp_input, temp_attention_mask, temp_label = custom_padding(padding_length, temp_input, temp_attention_mask, temp_label, tokenizer)


                        model_inputs["input_ids"][i] = torch.tensor(temp_input[-max_length:])
                        model_inputs["attention_mask"][i] = torch.tensor(temp_attention_mask[-max_length:])
                        labels["input_ids"][i] = torch.tensor(temp_label[-max_length:])

                
                # for i in range(len(correct_labels_extended)):
                #     sample_input_ids = model_inputs["input_ids"][i]
                    
                #     # print('sample_input_ids', sample_input_ids)
                #     # print('text', inputs[i])
                #     sample_attention_mask = model_inputs["attention_mask"][i]
                #     label_input_ids = labels["input_ids"][i][1:]
                #     label_attention_mask = labels["attention_mask"][i][1:]

                #     temp_input = sample_input_ids + label_input_ids
                #     print('len(temp_input)', len(temp_input))
                #     temp_attention_mask = sample_attention_mask + label_attention_mask
                #     label_ignore_pos = [tokenizer.pad_token_id] * len(sample_input_ids) + [-900] * len(label_input_ids)
                #     len_temp_input = len(temp_input)

                #     if len_temp_input >= max_length:
                #         temp_input = temp_input[-max_length:]
                #         temp_attention_mask = temp_attention_mask[-max_length:]
                #         label_ignore_pos = label_ignore_pos[-max_length:]
                #     else:
                #         temp_input = temp_input + [tokenizer.pad_token_id] * (max_length - len_temp_input)
                #         temp_attention_mask = temp_attention_mask + [0] * (max_length - len_temp_input)
                #         label_ignore_pos = label_ignore_pos + [tokenizer.pad_token_id] * (max_length - len_temp_input)
                    
                #     model_inputs["input_ids"][i] = temp_input
                #     model_inputs["attention_mask"][i] = temp_attention_mask

                #     temp_label = [None for _ in range(max_length)]
                #     index_for_label = 0
                #     for j in range(len(label_ignore_pos)):
                #         cur_ele = label_ignore_pos[j]
                #         if cur_ele == tokenizer.pad_token_id:
                #             temp_label[j] = -100
                #         else:
                #             temp_label[j] = label_input_ids[index_for_label]
                #             index_for_label += 1
                #     labels["input_ids"][i] = temp_label
                #     # labels["input_ids"][i] = copy.deepcopy(model_inputs["input_ids"][i])
                #     # labels["input_ids"][i] = [-100] * (len(sample_input_ids)+1)
                #     model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][-max_length:])
                #     model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][-max_length:])
                #     labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][-max_length:])
                #     # input_indices[i] = torch.tensor(input_indices[i])
                #     # correct_labels_extended[i] = torch.tensor(correct_labels_extended[i])

                model_inputs["labels"] = labels["input_ids"]
                model_inputs['input_indices'] = input_indices
                model_inputs["correct_labels"] = correct_labels_extended
                return model_inputs

            processed_dataset = {}
            processed_dataset['test'] = dataset['test'].map(
                tokenize_function,
                batched=True,
                # batch_size=50,
                batch_size=100000,
                num_proc=1,
                remove_columns=dataset["test"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                keep_in_memory=True,
            )
        elif cfg['data_name'] == 'obqa':
            '''
            {
                'id': '7-980',
                'question_stem': 'The sun is responsible for',
                'choices': {'text': ['puppies learning new tricks',
                'children growing up and getting old',
                'flowers wilting in a vase',
                'plants sprouting, blooming and wilting'],
                'label': ['A', 'B', 'C', 'D']},
                'answerKey': 'D'
            }
            '''
            # max_length = cfg[cfg['model_name']]['max_length']
            # cur_input_bsz = cfg['batch_size']
            # def tokenize_function(examples):
            #     batch_size = len(examples['answerKey'])
            #     correct_labels = examples['answerKey']
            #     # Convert each target to its numerical index
            #     correct_labels = [["A", "B", "C", "D"].index(choice) for choice in correct_labels]

            #     inputs = []
            #     labels = []
            #     correct_labels_extended = []
            #     input_indices = []

            #     for i in range(batch_size):
            #         num_choices = len(examples['choices'][i]['text'])
            #         inputs.extend([f"{examples['question_stem'][i]}"] * num_choices)
            #         labels.extend(examples['choices'][i]['text'])
            #         correct_labels_extended.extend([correct_labels[i]] * num_choices)
            #         input_indices.extend([i] * num_choices)

            #     tokenizer.truncation_side = 'left'
            #     model_inputs = tokenizer(inputs, max_length=max_length, padding="do_not_pad", truncation=True)
            #     tokenizer.truncation_side = 'right'
            #     labels = tokenizer(labels, max_length=max_length, padding="do_not_pad", truncation=True)

                
            #     for batch_index in range(0, len(correct_labels_extended), cur_input_bsz):
            #         # Determine the batch boundaries
            #         batch_end = min(batch_index + cur_input_bsz, len(correct_labels_extended))
            #         print('batch_index', batch_index)
            #         print('batch_end', batch_end)
            #         # Initialize a variable to track the maximum length of sequences in this batch
            #         max_length_in_batch = 0

            #         # First pass: calculate the maximum length in this batch
            #         for i in range(batch_index, batch_end):
            #             sample_input_ids = model_inputs["input_ids"][i]
            #             sample_attention_mask = model_inputs["attention_mask"][i]
            #             if labels["input_ids"][i][0] == tokenizer.bos_token_id:
            #                 label_input_ids = labels["input_ids"][i][1:]  # skip the first token
            #                 label_attention_mask = labels["attention_mask"][i][1:]
            #             else:
            #                 label_input_ids = labels["input_ids"][i]
            #                 label_attention_mask = labels["attention_mask"][i]

            #             # Combine the current input ids and label input ids
            #             temp_input = sample_input_ids + label_input_ids
            #             max_length_in_batch = max(max_length_in_batch, len(temp_input))

            #         # Second pass: adjust sequences to the max length in this batch
            #         for i in range(batch_index, batch_end):
            #             sample_input_ids = model_inputs["input_ids"][i]
            #             sample_attention_mask = model_inputs["attention_mask"][i]
            #             if labels["input_ids"][i][0] == tokenizer.bos_token_id:
            #                 label_input_ids = labels["input_ids"][i][1:]  # skip the first token
            #                 label_attention_mask = labels["attention_mask"][i][1:]
            #             else:
            #                 label_input_ids = labels["input_ids"][i]
            #                 label_attention_mask = labels["attention_mask"][i]

            #             temp_input = sample_input_ids + label_input_ids
            #             print('len(temp_input)', len(temp_input))
            #             temp_attention_mask = sample_attention_mask + label_attention_mask
            #             temp_label = [-100] * len(sample_input_ids) + label_input_ids

            #             # Truncate or pad sequences based on the batch's maximum length
            #             if len(temp_input) > max_length_in_batch:
            #                 temp_input = temp_input[-max_length_in_batch:]
            #                 temp_attention_mask = temp_attention_mask[-max_length_in_batch:]
            #                 temp_label = temp_label[-max_length_in_batch:]
            #             else:
            #                 padding_length = max_length_in_batch - len(temp_input)
            #                 temp_input, temp_attention_mask, temp_label = custom_padding(padding_length, temp_input, temp_attention_mask, temp_label, tokenizer)


            #             model_inputs["input_ids"][i] = torch.tensor(temp_input[-max_length:])
            #             model_inputs["attention_mask"][i] = torch.tensor(temp_attention_mask[-max_length:])
            #             labels["input_ids"][i] = torch.tensor(temp_label[-max_length:])
            #         print('max_batch', max_length_in_batch)

            #     model_inputs["labels"] = labels["input_ids"]
            #     model_inputs['input_indices'] = input_indices
            #     model_inputs["correct_labels"] = correct_labels_extended
            #     return model_inputs

            max_length = cfg[cfg['model_name']]['max_length']
            cur_input_bsz = cfg['batch_size']
            def tokenize_function(examples):
                batch_size = len(examples['answerKey'])
                correct_labels = examples['answerKey']
                # Convert each target to its numerical index
                correct_labels = [["A", "B", "C", "D"].index(choice) for choice in correct_labels]

                inputs = []
                labels = []
                correct_labels_extended = []
                input_indices = []

                for i in range(batch_size):
                    num_choices = len(examples['choices'][i]['text'])
                    inputs.extend([f"{examples['question_stem'][i]}"] * num_choices)
                    labels.extend(examples['choices'][i]['text'])
                    correct_label = [0] * num_choices
                    correct_label[correct_labels[i]] = 1
                    correct_labels_extended.extend(correct_label)
                    input_indices.extend([i] * num_choices)

                tokenizer.truncation_side = 'left'
                model_inputs = tokenizer(inputs, max_length=max_length, padding="do_not_pad", truncation=True)
                tokenizer.truncation_side = 'right'
                labels = tokenizer(labels, max_length=max_length, padding="do_not_pad", truncation=True)

                for i in range(len(correct_labels_extended)):
                    if labels["input_ids"][i][0] == tokenizer.bos_token_id:
                        labels["input_ids"][i] = labels["input_ids"][i][1:]  # skip the first token
                        labels["attention_mask"][i] = labels["attention_mask"][i][1:]


                # temp_input_indices = copy.deepcopy(input_indices)
                # sequence_lengths = [len(inp_ids) + len(lbl_ids) for inp_ids, lbl_ids in zip(model_inputs["input_ids"], labels["input_ids"])]

                # # Sort indices based on sequence lengths for batching
                # sorted_indices = sorted(range(len(sequence_lengths)), key=lambda k: sequence_lengths[k])
                # print('sorted_indicesdataset', sorted_indices)
                # # Sort all inputs and labels based on sorted indices
                # model_inputs = {key: [val[idx] for idx in sorted_indices] for key, val in model_inputs.items()}
                # labels = {key: [val[idx] for idx in sorted_indices] for key, val in labels.items()}
                # correct_labels_extended = [correct_labels_extended[idx] for idx in sorted_indices]
                # input_indices = [input_indices[idx] for idx in sorted_indices]

                # print('temp_input_indices', temp_input_indices)
                print('input_indices', input_indices)
                print('correct_labels_extended', correct_labels_extended)

                for batch_index in range(0, len(correct_labels_extended), cur_input_bsz):
                    # Determine the batch boundaries
                    batch_end = min(batch_index + cur_input_bsz, len(correct_labels_extended))
                    print('batch_index', batch_index)
                    print('batch_end', batch_end)
                    # Initialize a variable to track the maximum length of sequences in this batch
                    max_length_in_batch = 0

                    # First pass: calculate the maximum length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        # Combine the current input ids and label input ids
                        temp_input = sample_input_ids + label_input_ids
                        max_length_in_batch = max(max_length_in_batch, len(temp_input))
                    #     print('len(temp_input)', len(temp_input))
                    # print('max_batch', max_length_in_batch)
                    # Second pass: adjust sequences to the max length in this batch
                    for i in range(batch_index, batch_end):
                        sample_input_ids = model_inputs["input_ids"][i]
                        sample_attention_mask = model_inputs["attention_mask"][i]
                        label_input_ids = labels["input_ids"][i]
                        label_attention_mask = labels["attention_mask"][i]

                        temp_input = sample_input_ids + label_input_ids
                        print('len(temp_input)', len(temp_input))
                        temp_attention_mask = sample_attention_mask + label_attention_mask
                        temp_label = [-100] * len(sample_input_ids) + label_input_ids

                        # Truncate or pad sequences based on the batch's maximum length
                        if len(temp_input) > max_length_in_batch:
                            temp_input = temp_input[-max_length_in_batch:]
                            temp_attention_mask = temp_attention_mask[-max_length_in_batch:]
                            temp_label = temp_label[-max_length_in_batch:]
                        else:
                            padding_length = max_length_in_batch - len(temp_input)
                            temp_input, temp_attention_mask, temp_label = custom_padding(padding_length, temp_input, temp_attention_mask, temp_label, tokenizer)


                        model_inputs["input_ids"][i] = torch.tensor(temp_input[-max_length:])
                        model_inputs["attention_mask"][i] = torch.tensor(temp_attention_mask[-max_length:])
                        labels["input_ids"][i] = torch.tensor(temp_label[-max_length:])
                    print('max_batch', max_length_in_batch)

                model_inputs["labels"] = labels["input_ids"]
                model_inputs['input_indices'] = input_indices
                model_inputs["correct_labels"] = correct_labels_extended
                return model_inputs
            
            processed_dataset = {}
            processed_dataset['test'] = dataset['test'].map(
                tokenize_function,
                batched=True,
                batch_size=100000,
                num_proc=1,
                remove_columns=dataset["test"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                keep_in_memory=True,
            )
        else:
            raise ValueError('Not valid data name')
        cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
        cfg['target_size'] = len(tokenizer)
    elif cfg['task_name'] in ['ic']:
        processed_dataset = dataset
        cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
        cfg['target_size'] = processed_dataset['train'].target_size
    else:
        raise ValueError('Not valid task name')
    return processed_dataset


def make_batchnorm_stats(dataset_, model, tag):
    dataset_ = copy.deepcopy(dataset_)
    model = copy.deepcopy(model)
    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                v.momentum = None
                v.track_running_stats = True
                v.register_buffer('running_mean', torch.zeros(v.num_features, device=cfg['device']))
                v.register_buffer('running_var', torch.ones(v.num_features, device=cfg['device']))
                v.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=cfg['device']))
        transform = dataset.Compose([transforms.ToTensor(), transforms.Normalize(*data_stats[cfg['data_name']])])
        dataset_.transform = transform
        data_loader = make_data_loader({'train': dataset_}, None, tag, shuffle={'train': False})['train']
        model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            model(**input)
    return model
