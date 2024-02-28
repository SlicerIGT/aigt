from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    """Adapted from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/lr_scheduler/polylr.py"""
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, last_step: int = -1):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.last_step = last_step
        super().__init__(optimizer, last_step, False)

    def step(self, epoch=None):
        self.last_step += 1
        new_lr = self.initial_lr * (1 - self.last_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class LinearWarmupWrapper(_LRScheduler):
    """Wrapper for a PyTorch scheduler to add a linear LR warmup."""
    def __init__(self, optimizer, scheduler, initial_lr, warmup_steps, last_step=-1):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.last_step = last_step
        super().__init__(optimizer, last_step, False)
    
    def step(self, epoch=None):
        self.last_step += 1
        if self.last_step <= self.warmup_steps:
            warmup_factor = min(1.0, (self.last_step + 1) / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * warmup_factor
        else:
            self.scheduler.step()
