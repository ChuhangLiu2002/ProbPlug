"""
Utilization functions
"""

import os
import random
import sys
# from torch.optim import CosineAnnealingWarmRestarts
import numpy as np
import torch
import yaml
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class OptimConfig:
    optimizer: str
    epochs: int
    amsgrad: str
    base_lr: float
    lr_min: float
    betas: List[float]
    weight_decay: float
    scheduler: str
    steps_per_epoch: int

@dataclass
class Config:
    config_file: str = "config.yaml"  # 默认配置文件名
    
    def __post_init__(self):
        # 从YAML文件加载配置
        with open(self.config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        self.fusemodel = data['fusemodel']
        self.train_dir = data['train_dir']
        self.val_dir = data['val_dir']
        self.model_name = data['model_name']
        self.target_layers = data['target_layers']
        self.seq_pool = data['seq_pool']
        self.select_seq = data['select_seq']
        self.feature_aggregation = data['feature_aggregation']
        self.torch_dtype = data['torch_dtype']
        self.num_classes = data['num_classes']
        self.batch_size = data['batch_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = data['save_dir']
        self.resume = data['resume']
        
        # 嵌套的优化器配置
        optim_data = data['optim_config']
        self.optim_config = OptimConfig(
            optimizer=optim_data['optimizer'],
            epochs=optim_data['epochs'],
            amsgrad=optim_data['amsgrad'],
            base_lr=optim_data['base_lr'],
            lr_min=optim_data['lr_min'],
            betas=optim_data['betas'],
            weight_decay=optim_data['weight_decay'],
            scheduler=optim_data['scheduler'],
            steps_per_epoch=optim_data['steps_per_epoch']
        )


def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError('invalid truth value {}'.format(val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras-style"""
    return 1. / (1. + decay * step)


class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """SGD with restarts scheduler"""
    def __init__(self, optimizer, T0, T_mul, eta_min, last_epoch=-1):
        self.Ti = T0
        self.T_mul = T_mul
        self.eta_min = eta_min

        self.last_restart = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T_cur = self.last_epoch - self.last_restart
        if T_cur >= self.Ti:
            self.last_restart = self.last_epoch
            self.Ti = self.Ti * self.T_mul
            T_cur = 0

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + np.cos(np.pi * T_cur / self.Ti)) / 2
            for base_lr in self.base_lrs
        ]


def _get_optimizer(model_parameters, optim_config):
    """Defines optimizer according to the given config"""
    optimizer_name = optim_config.optimizer

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=optim_config.base_lr,
                                    momentum=optim_config.momentum,
                                    weight_decay=optim_config.weight_decay,
                                    nesterov=optim_config.nesterov)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=optim_config.base_lr,
                                     betas=optim_config.betas,
                                     weight_decay=optim_config.weight_decay,
                                     amsgrad=str_to_bool(
                                         optim_config.amsgrad))
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model_parameters,
                                      lr=optim_config.base_lr,
                                      betas=optim_config.betas,
                                      weight_decay=optim_config.weight_decay,
                                      amsgrad=str_to_bool(
                                      optim_config.amsgrad))
    else:
        print('Un-known optimizer', optimizer_name)
        sys.exit()

    return optimizer


def _get_scheduler(optimizer, optim_config):
    """
    Defines learning rate scheduler according to the given config
    """
    if optim_config.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config.milestones,
            gamma=optim_config.lr_decay)

    elif optim_config.scheduler == 'sgdr':
        scheduler = SGDRScheduler(optimizer, optim_config.T0,
                                  optim_config.Tmult,
                                  optim_config.lr_min)

    elif optim_config.scheduler == 'cosine':
        total_steps = optim_config.epochs * \
            optim_config.steps_per_epoch

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                optim_config.lr_min / optim_config.base_lr))
        # scheduler = CosineAnnealingWarmRestarts(optimizer, optim_config['T_0'], optim_config['T_mult'])

    elif optim_config.scheduler == 'keras_decay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: keras_decay(step))
    else:
        scheduler = None
    return scheduler

def create_optimizer(model_parameters, optim_config):
    """Defines an optimizer and a scheduler"""
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, config = None):
    """ 
    set initial seed for reproduction
    """
    if config is None:
        raise ValueError("config should not be None")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = str_to_bool(config.cudnn_deterministic_toggle)
        torch.backends.cudnn.benchmark = str_to_bool(config.cudnn_benchmark_toggle)
