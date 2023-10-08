import os
import math
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from torch.optim.lr_scheduler import _LRScheduler  

'''
self.optimizer = utility.make_optimizer(args, self.model)
'''
def dict_to_class(**dict):
    class _dict_to_class:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return _dict_to_class(**dict)

optimizer_params = {
    "lr": 4e-4,
    "decay_type": 'multi step',
    "decay": [200, 200, 200, 200, 200], 
    "step_size": 200,
    "gamma": 0.5,
    "optimizer": 'ADAM',
    "momentum": 0.9,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "weight_decay": 0.0,
    "gclip": 0.0
}
optimizer_args = dict_to_class(**optimizer_params)
       
def make_optimizer(target, args=optimizer_args):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay))
    if args.decay_type == 'multi step':
        kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
        scheduler_class = lrs.MultiStepLR
    elif args.decay_type == 'step':
        kwargs_scheduler = {'step_size': args.step_size, 'gamma': args.gamma}
        scheduler_class = lrs.StepLR
    elif args.decay_type == 'cosine':
        T_period = [args.step_size * i for i in range(1, args.epoch // args.step_size + 1)]
        restarts = [args.step_size * i for i in range(1, args.epoch // args.step_size + 1)]
        restart_weights = [1] * (args.epoch // args.step_size)

        kwargs_scheduler = {'T_period': T_period, 'eta_min': 1e-7, 'restarts': restarts, 'weights': restart_weights}
        scheduler_class = CosineAnnealingLR_Restart


    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.decay_type == 'cosine':
        # cosine annealing restart
        ## need more to prevent error
        T_period = [args.step_size * i for i in range(1, args.epoch // args.step_size + 1)]
        restarts = [args.step_size * i for i in range(1, args.epoch // args.step_size + 1)]
        restart_weights = [1] * (args.epoch // args.step_size)

        scheduler = CosineAnnealingLR_Restart(my_optimizer, T_period, eta_min=1e-7, restarts=restarts,
                                            weights=restart_weights)

    return scheduler

class CosineAnnealingLR_Restart(_LRScheduler):
    """
    ref:https://github.com/zhaohengyuan1/PAN/blob/a20974545cf011c386d728739d091c39e23d0686/codes/models/lr_scheduler.py
    ref: pytorch_CosineAnnealingLR_doc  https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
    """
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]
        
        
if __name__ == "__main__":
    pass
    # model = GolemModel(5, 1, 1, )
    # optimizer = make_optimizer(model)
    # save_dir = 'path/to/model'
    # last_epoch = 100
    # optimizer.load(save_dir, epoch=last_epoch)
    # epoch = optimizer.get_last_epoch() + 1
    # lr = optimizer.get_lr()
    # optimizer.zero_grad()
    # optimizer.step()
    # optimizer.save(save_dir)
    
    # optimizer.schedule()
    # optimizer.get_last_epoch()
    