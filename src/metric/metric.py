import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from config import cfg
from module import recur


def make_metric(metric_name, tokenizer):
    if cfg['task_name'] == 'clm':
        if cfg['data_name'] in ['wikitext', 'c4', 'ptb']:
            pivot = float('inf')
            pivot_direction = 'down'
            pivot_name = 'Perplexity'
            metric_name['train'].extend(['Perplexity'])
            metric_name['test'].extend(['Perplexity'])
        else:
            raise ValueError('Not valid data name')
    elif cfg['task_name'] == 'ic':
        pivot = -float('inf')
        pivot_direction = 'up'
        pivot_name = 'Accuracy'
        for k in metric_name:
            metric_name[k].extend(['Accuracy'])
    elif cfg['task_name'] == 'csr':
        if cfg['data_name'] in ['boolq', 'piqa', 'siqa', 'arc', 'hellaswag', 'winogrande', 'obqa']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'Accuracy'
            for k in metric_name:
                metric_name[k].extend(['CsrAccuracy'])
                metric_name[k].extend(['CsrAccuracyNorm'])
        else:
            raise ValueError('Not valid data name')
    else:
        raise ValueError('Not valid task name')
    metric = Metric(metric_name, pivot, pivot_direction, pivot_name, tokenizer)
    return metric


def Loss(output):
    loss = output.item()
    return loss


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        if target.dtype != torch.int64:
            target = (target.topk(1, -1, True, True)[1]).view(-1)
        batch_size = torch.numel(target)
        pred_k = output.topk(topk, -1, True, True)[1]
        correct_k = pred_k.eq(target.unsqueeze(-1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def RMSE(output, target):
    with torch.no_grad():
        rmse = F.mse_loss(output, target).sqrt().item()
    return rmse

class Perplexity:
    def __init__(self):
        self.loss_list = []
        return
    
    def add(self, input, output):
        loss = output['loss'].item()
        self.loss_list.append(loss)
        return
       
    def __call__(self, *args, **kwargs):
        return torch.exp(torch.tensor(np.array(self.loss_list).mean())).item()

class CsrAccuracy:
    def __init__(self):
        self.acc_output_for_one_question = defaultdict(list)
        self.acc_correct_labels_for_one_question = defaultdict(list)
        self.average = []
    
    def add(self, input, output):
        lm_logits = output['target']
        labels = input['target']

        bsz = lm_logits.size(0)
        
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        label_length_per_sample = torch.sum(shift_labels != -100, dim=1)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        shift_labels = shift_labels.to(shift_logits.device)
        print('shift_logitsdevice', shift_logits.device)
        print('shift_labelsdevice', shift_labels.device)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(bsz, -1)
        loss_per_sample = loss.sum(dim=1)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        self.average.append(loss.item())

        # acc
        for i in range(input['input_indices'].shape[0]):
            self.acc_output_for_one_question[input['input_indices'][i].item()].append(loss_per_sample[i].item())
            self.acc_correct_labels_for_one_question[input['input_indices'][i].item()].append(int(input['correct_labels'][i].item()))
        

    def __call__(self, *args, **kwargs):    
        total_acc = 0
        for key in self.acc_output_for_one_question:
            # argmin for positive loss
            correct_index = next((i for i, item in enumerate(self.acc_correct_labels_for_one_question[key]) if item == 1), None)
            acc = 1 if np.argmin(self.acc_output_for_one_question[key]) == correct_index else 0
            total_acc += acc

        ppl = np.exp(np.mean(self.average))
        print('pplforcsr', ppl)
        return (total_acc / len(self.acc_output_for_one_question)) * 100


class CsrAccuracyNorm:
    def __init__(self):
        self.acc_norm_output_for_one_question = defaultdict(list)
        self.acc_norm_correct_labels_for_one_question = defaultdict(list)
        self.average = []
        
    def add(self, input, output):
        lm_logits = output['target']
        labels = input['target']

        bsz = lm_logits.size(0)
        
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        label_length_per_sample = torch.sum(shift_labels != -100, dim=1)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(bsz, -1)
        loss_per_sample = loss.sum(dim=1)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        self.average.append(loss.item())
        
        # accnorm
        # print('loss_per_sample', loss_per_sample)
        # print('label_length_per_sample', label_length_per_sample)
        for i in range(input['input_indices'].shape[0]):
            self.acc_norm_output_for_one_question[input['input_indices'][i].item()].append(loss_per_sample[i].item()/label_length_per_sample[i].item())
            self.acc_norm_correct_labels_for_one_question[input['input_indices'][i].item()].append(int(input['correct_labels'][i].item()))

    def __call__(self, *args, **kwargs):    
        total_acc = 0
        for key in self.acc_norm_output_for_one_question:
            # argmin for positive loss
            correct_index = next((i for i, item in enumerate(self.acc_norm_correct_labels_for_one_question[key]) if item == 1), None)
            acc = 1 if np.argmin(self.acc_norm_output_for_one_question[key]) == correct_index else 0
            total_acc += acc

        ppl = np.exp(np.mean(self.average))
        print('pplforcsr', ppl)

        return (total_acc / len(self.acc_norm_output_for_one_question)) * 100
    
class Metric:
    def __init__(self, metric_name, pivot, pivot_direction, pivot_name, tokenizer):
        self.pivot, self.pivot_name, self.pivot_direction = pivot, pivot_name, pivot_direction
        self.metric_name = metric_name
        self.metric = self.make_metric(metric_name, tokenizer)

    def make_metric(self, metric_name, tokenizer):
        metric = defaultdict(dict)
        for split in metric_name:
            for m in metric_name[split]:
                if m == 'Loss':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: recur(Loss, output['loss']))}
                elif m == 'Perplexity':
                    # metric[split][m] = {'mode': 'batch', 'metric': (lambda input,
                    #                                                        output: recur(Perplexity, output['loss']))}
                    metric[split][m] = {'mode': 'full', 'metric': Perplexity()}
                elif m == 'CsrAccuracy':
                    metric[split][m] = {'mode': 'full', 'metric': CsrAccuracy()}
                elif m == 'CsrAccuracyNorm':
                    metric[split][m] = {'mode': 'full', 'metric': CsrAccuracyNorm()}
                else:
                    raise ValueError('Not valid metric name')
        return metric

    def add(self, split, input, output):
        for metric_name in self.metric_name[split]:
            if self.metric[split][metric_name]['mode'] == 'full':
                self.metric[split][metric_name]['metric'].add(input, output)
        return

    def evaluate(self, split, mode, input=None, output=None, metric_name=None):
        metric_name = self.metric_name if metric_name is None else metric_name
        evaluation = {}
        for metric_name_ in metric_name[split]:
            if self.metric[split][metric_name_]['mode'] == mode:
                evaluation[metric_name_] = self.metric[split][metric_name_]['metric'](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return

    def load_state_dict(self, state_dict):
        self.pivot = state_dict['pivot']
        self.pivot_name = state_dict['pivot_name']
        self.pivot_direction = state_dict['pivot_direction']
        return

    def state_dict(self):
        return {'pivot': self.pivot, 'pivot_name': self.pivot_name, 'pivot_direction': self.pivot_direction}
