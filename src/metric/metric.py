import torch
import torch.nn.functional as F
import evaluate
from collections import defaultdict
from config import cfg
from module import recur


def make_metric(metric_name, tokenizer):
    if cfg['task_name'] == 'clm':
        if cfg['data_name'] in ['dolly']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'ROUGE'
            metric_name['train'].extend(['Perplexity'])
            metric_name['test'].extend(['ROUGE'])
        else:
            raise ValueError('Not valid data name')
    elif cfg['task_name'] == 's2s':
        if cfg['data_name'] in ['fpb', 'wikisql', 'samsum', 'e2enlg', 'webnlg', 'dart']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'ROUGE'
            for k in metric_name:
                metric_name[k].extend(['Accuracy'])
            metric_name['test'].extend(['ROUGE'])
        else:
            raise ValueError('Not valid data name')
    elif cfg['task_name'] == 'sc':
        if cfg['data_name'] in ['glue']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'GLUE'
            metric_name['test'].extend(['GLUE'])
        else:
            raise ValueError('Not valid data name')
    elif cfg['task_name'] == 'ic':
        pivot = -float('inf')
        pivot_direction = 'up'
        pivot_name = 'Accuracy'
        for k in metric_name:
            metric_name[k].extend(['Accuracy'])
    elif cfg['task_name'] == 't2i':
        if cfg['data_name'] in ['dreambooth']:
            pivot = float('inf')
            pivot_direction = 'down'
            pivot_name = 'Loss'
        else:
            raise ValueError('Not valid data name')
    else:
        raise ValueError('Not valid task name')
    metric = Metric(metric_name, pivot, pivot_direction, pivot_name, tokenizer)
    return metric


def Loss(output):
    loss = output.item()
    return loss


def Perplexity(output):
    ppl = output.exp().item()
    return ppl


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


class GLUE:
    def __init__(self, subset_name):
        self.metric = evaluate.load('glue', subset_name)
        self.subset_name = subset_name

    def add(self, input, output):
        if self.subset_name in ['stsb']:
            predictions = output['target']
        else:
            predictions = output['target'].argmax(dim=-1)
        references = input['target']
        self.metric.add_batch(predictions=predictions, references=references)
        return

    def __call__(self, *args, **kwargs):
        glue = self.metric.compute()
        metric_name = list(glue.keys())[0]
        glue = glue[metric_name]
        return glue


class ROUGE:
    def __init__(self, tokenizer, split_metric):
        self.split_metric = split_metric
        if cfg['dist_mode'] in ['alone', 'col'] and self.split_metric:
            self.metric = [evaluate.load('rouge') for _ in range(cfg['num_split'])]
        else:
            self.metric = evaluate.load('rouge')
        self.tokenizer = tokenizer

    def decode(self, generate, target):
        generate = generate[:, -cfg['max_new_tokens']:]
        target[target < 0] = cfg['pad_token_id']
        generate = self.tokenizer.batch_decode(generate.detach().cpu().numpy(), skip_special_tokens=True)
        target = self.tokenizer.batch_decode(target.detach().cpu().numpy(), skip_special_tokens=True)
        return generate, target

    def add(self, input, output):
        if cfg['dist_mode'] in ['alone', 'col'] and self.split_metric:
            for i in range(cfg['num_split']):
                generate_i = output['generate'][i]
                if generate_i is None:
                    continue
                target_i = input['target'][i]
                generate_i, target_i = self.decode(generate_i, target_i)
                self.metric[i].add_batch(predictions=generate_i, references=target_i)
        else:
            generate = output['generate']
            target = input['target']
            generate, target = self.decode(generate, target)
            self.metric.add_batch(predictions=generate, references=target)
        return

    def __call__(self, *args, **kwargs):
        if cfg['dist_mode'] in ['alone', 'col'] and self.split_metric:
            rouge = []
            for i in range(cfg['num_split']):
                rouge_i = self.metric[i].compute()['rougeL']
                rouge.append(rouge_i)
        else:
            rouge = self.metric.compute()['rougeL']
        return rouge


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
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input,
                                                                           output: recur(Perplexity, output['loss']))}
                elif m == 'Accuracy':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(Accuracy, output['target'], input['target']))}
                elif m == 'RMSE':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(RMSE, output['target'], input['target']))}
                elif m == 'ROUGE':
                    metric[split][m] = {'mode': 'full', 'metric': ROUGE(tokenizer, cfg['split_metric'])}
                elif m == 'GLUE':
                    metric[split][m] = {'mode': 'full', 'metric': GLUE(cfg['hf_subset_name'])}
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
