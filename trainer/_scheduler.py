
from typing import Union, Type

import torch.nn as nn
import torch.optim

from utils import CONFIG

scheduler_dict = {
    'exponential' : torch.optim.lr_scheduler.ExponentialLR,
    'multistep' : torch.optim.lr_scheduler.MultiStepLR,
    'cosine' : torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosinewarmup' : torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}

def _get_scheduler(
        optimizer : nn.Module,
        cfg : Type[CONFIG],
        **kwargs
    ) -> Union[nn.Module, None] :

    if cfg.SCHEDULER not in scheduler_dict :
        print("Invaild Scheduler, Set scheduler as None...")
        return None
    
    return scheduler_dict[cfg.SCHEDULER]( optimizer = optimizer,
                                    **cfg.SCHEDULER_ARGS )
