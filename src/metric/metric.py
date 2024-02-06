import torch
import numpy as np
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
        elif cfg['data_name'] in ['wikitext']:
            pivot = float('inf')
            pivot_direction = 'down'
            pivot_name = 'Perplexity'
            metric_name['train'].extend(['Perplexity'])
            metric_name['test'].extend(['Perplexity'])
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
    elif cfg['task_name'] == 'csr':
        if cfg['data_name'] in ['boolq', 'piqa', 'siqa', 'arc', 'aec','hellaswag', 'winogrande', 'obqa']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'Accuracy'
            for k in metric_name:
                metric_name[k].extend(['CsrAccuracy'])
        else:
            raise ValueError('Not valid data name')
    else:
        raise ValueError('Not valid task name')
    metric = Metric(metric_name, pivot, pivot_direction, pivot_name, tokenizer)
    return metric


def Loss(output):
    loss = output.item()
    return loss


# def Perplexity(output):
#     ppl = torch.exp(output).item()
#     return ppl


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
        # print('self.loss_list', self.loss_list, torch.tensor(self.loss_list))
        # print('Perplexity', torch.mean(torch.tensor(self.loss_list)))
        # print('res', torch.exp(torch.mean(torch.tensor(self.loss_list))))
        # print('0', self.loss_list)
        # print('a', np.array(self.loss_list).mean())
        # print('res', torch.exp(torch.tensor(np.array(self.loss_list).mean())))
        return torch.exp(torch.tensor(np.array(self.loss_list).mean())).item()
        # return torch.exp(torch.mean(torch.tensor(self.loss_list))).item()


class CsrAccuracy:
    def __init__(self):
        self.output_for_one_question = defaultdict(list)
        self.correct_labels_for_one_question = defaultdict(list)
        pass
    
    def add(self, input, output):
        # generate = output['generate'].detach().cpu()
        # scores = output['scores']
        # target = input['target'].detach().cpu()

        # generate = generate[:, -cfg['max_new_tokens']:]
        # def tuple_of_tensors_to_tensor(tuple_of_tensors):
        #     return torch.stack(list(tuple_of_tensors), dim=0)
        
        # scores = tuple_of_tensors_to_tensor(scores)
        # scores = scores.view(generate.shape[0], -1, scores.shape[-1]).detach().cpu()
        # a = scores.shape
        # scores = scores[:, -cfg['max_new_tokens']:, :]
        # # scores = F.log_softmax(scores, dim=-1)
        # target[target < 0] = cfg['pad_token_id']
        # target = target[:, -cfg['max_new_tokens']:]

        # non_pad_mask = target != cfg['pad_token_id']
        lm_logits = output['target']
        labels = input['target']
        bsz = lm_logits.size(0)
        
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        seq_len = shift_logits.size(1)
        # print('shift_logits', shift_logits)
        # print('shift_labels', shift_labels)
        # shift_logits = F.log_softmax(shift_logits, dim=-1)
        # inplen = shift_logits.size(1)
        # contlen = shift_labels.size(1)
        # logits = logits[inplen - contlen : inplen].unsqueeze(
        #     0
        # ) 

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # print('loss', loss, loss.shape)
        loss = loss.view(bsz, -1)
        # print('loss after view', loss, loss.shape)
        loss_per_sample = loss.sum(dim=1)
        # print('loss_per_sample', loss_per_sample)


        # print('loss persample', loss_per_sample)
        # print('CsrAccuracy', loss_per_sample)

        for i in range(input['input_indices'].shape[0]):
            self.output_for_one_question[input['input_indices'][i].item()].append(loss_per_sample[i].item())
            self.correct_labels_for_one_question[input['input_indices'][i].item()].append(input['correct_labels'][i].item())

        # a = 5
    def __call__(self, *args, **kwargs):
        # print('self.output_for_one_question', self.output_for_one_question)
        # print('self.correct_labels_for_one_question', self.correct_labels_for_one_question)
        total_acc = 0
        for key in self.output_for_one_question:
            # argmin for positive loss
            acc = 1 if np.argmin(self.output_for_one_question[key]) == self.correct_labels_for_one_question[key][0] else 0
            total_acc += acc

        return (total_acc / len(self.output_for_one_question)) * 100

class ROUGE:
    def __init__(self, tokenizer, split_metric):
        self.split_metric = split_metric
        self.metric = evaluate.load('rouge')
        self.tokenizer = tokenizer

    def decode(self, generate, target):
        generate = generate[:, -cfg['max_new_tokens']:]
        target[target < 0] = cfg['pad_token_id']
        generate = self.tokenizer.batch_decode(generate.detach().cpu().numpy(), skip_special_tokens=True)
        target = self.tokenizer.batch_decode(target.detach().cpu().numpy(), skip_special_tokens=True)
        return generate, target

    def add(self, input, output):
        generate = output['generate']
        target = input['target']
        generate, target = self.decode(generate, target)
        self.metric.add_batch(predictions=generate, references=target)
        return

    def __call__(self, *args, **kwargs):
        rouge = self.metric.compute()['rougeL']
        return rouge

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
                elif m == 'CsrAccuracy':
                    metric[split][m] = {'mode': 'full', 'metric': CsrAccuracy()}
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
