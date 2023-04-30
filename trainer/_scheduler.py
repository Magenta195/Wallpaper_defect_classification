
from typing import Union

import torch.nn as nn
import torch.optim

from .utils import CONFIG

scheduler_dict = {
    'exponential' : torch.optim.lr_scheduler.ExponentialLR,
    'multistep' : torch.optim.lr_scheduler.MultiStepLR,
    'adamw' : torch.optim.lr_scheduler.CosineAnnealingLR
}

def _get_scheduler(
        scheduler_name : str,
        optimizer : nn.Module,
        **kwargs
    ) -> Union[nn.Module, None] :

    if scheduler_name not in scheduler_dict :
        print("Invaild Scheduler, Set scheduler as None...")
        return None
    
    return scheduler_dict[scheduler_name]( optimizer = optimizer,
                                    **kwargs )
