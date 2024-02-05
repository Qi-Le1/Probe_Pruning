import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model
from torchvision import transforms
from transformers import get_linear_schedule_with_warmup
from config import cfg
from diffusers import DDPMScheduler
from .huggingface import make_hf_model

from module import TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING


def make_model(model_name, sub_model_name=None):
    if cfg['task_name'] in ['s2s', 'sc', 'clm', 'csr', 't2i']:
        model, tokenizer = make_hf_model(model_name, sub_model_name)
        # base_model_name_or_path = model.__dict__.get("name_or_path", None)
        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        model_type = model_config["model_type"]
        cfg['model_type'] = model_type

        
    else:
        model = eval('model.{}()'.format(model_name))
        model = model.to(cfg['device'])
        tokenizer = None
        cfg['model_type'] = model_name
    return model, tokenizer

def make_prune_model(model):
    from .llama_eri import LlamaEriModel
    if 'llama' in cfg['model_name']:
        model = LlamaEriModel(model)
    else:
        raise ValueError('Not valid model name')
    return model

def make_calibration_prune_model(model):
    from .llama_ewi import LlamaEwiModel
    if 'llama' in cfg['model_name']:
        model = LlamaEwiModel(model)
    else:
        raise ValueError('Not valid model name')
    return model

def make_loss(output, input):
    if 'target' in input:
        loss = loss_fn(output['target'], input['target'])
    else:
        return
    return loss


def loss_fn(output, target, reduction='mean'):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = kld_loss(output, target, reduction=reduction)
    return loss


def cross_entropy_loss(output, target, reduction='mean'):
    if target.dtype != torch.int64:
        target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, reduction=reduction)
    return ce


def kld_loss(output, target, reduction='batchmean'):
    kld = F.kl_div(F.log_softmax(output, dim=-1), target, reduction=reduction)
    return kld


def mse_loss(output, target, reduction='mean'):
    mse = F.mse_loss(output, target, reduction=reduction)
    return mse


def init_param(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.zero_()
    return m


def make_optimizer(parameters, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(parameters, lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(parameters, lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                               weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                                weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(parameters, lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


class NoOpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(NoOpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = NoOpScheduler(optimizer)
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_steps']['train'] *
                                                                          cfg[cfg['model_name']]['num_epochs'],
                                                         eta_min=0)
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    elif cfg[tag]['scheduler_name'] == 'LinearAnnealingLR':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
            cfg['num_steps']['train'] * cfg[cfg['model_name']]['num_epochs'] * cfg[tag]['warmup_ratio']),
                                                    num_training_steps=cfg['num_steps']['train'] *
                                                                       cfg[cfg['model_name']]['num_epochs'])
    elif cfg[tag]['scheduler_name'] == 'ConstantLR':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=cfg[tag]['factor'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def make_noise_scheduler(tag):
    if 'noise_scheduler_name' not in cfg[tag]:
        raise ValueError('Not valid noise scheduler name')

    if cfg[tag]['noise_scheduler_name'] == 'DDPM':
        noise_scheduler = DDPMScheduler(
            beta_start=cfg[tag]['beta_start'],
            beta_end=cfg[tag]['beta_end'],
            beta_schedule=cfg[tag]['beta_schedule'],
            num_train_timesteps=cfg[tag]['num_train_timesteps'],
        )
    else:
        raise ValueError('Not valid noise scheduler name')
    return noise_scheduler

def freeze_model(model):
    if cfg['ft_name'] == 'cola':
        for n, p in model.named_parameters():
            p.requires_grad = False
    return


def unfreeze_model(model):
    if cfg['ft_name'] == 'cola':
        for n, p in model.named_parameters():
            p.requires_grad = True
    return
