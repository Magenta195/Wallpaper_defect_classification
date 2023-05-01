
from typing import Type

import torch.nn as nn
import torch.optim

from .utils import CONFIG

optimizer_dict = {
    'Adam' : torch.optim.Adam,
    'SGD' : torch.optim.SGD,
    'Adamw' : torch.optim.AdamW
}

def _get_optimizer(
        model_param : any,
        cfg : Type[CONFIG],
        **kwargs
    ) -> nn.Module :

    if cfg.OPTIMIZER not in optimizer_dict :
        raise NotImplementedError("Invaild Optimizer")
    
    return optimizer_dict[cfg.OPTIMIZER]( params = model_param,
                                    lr = cfg.LEARNING_RATE,
                                    **kwargs )
