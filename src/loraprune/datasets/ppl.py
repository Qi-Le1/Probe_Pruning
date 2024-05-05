import torch
import numpy as np
from tqdm import tqdm
import random

from datasets import load_dataset
from torch.utils.data.dataset import Dataset


def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size=4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size=batch_size)
        ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric


@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    # print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


def get_wikitext2(seq_len, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return traindata, testdata


def get_ptb(seq_len, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    return traindata, valdata


class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)


def get_loaders(name, tokenizer, seq_len=2048, batch_size=8):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        train_data, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader
