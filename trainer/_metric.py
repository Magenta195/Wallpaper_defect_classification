from typing import Type

import torch.nn as nn
from sklearn.metrics import f1_score
from .utils import CONFIG

loss_dict = {
    'celoss' : nn.CrossEntropyLoss,
#    'focalloss' : ,
}

score_dict = {
    'f1score' : f1_score
}

def _get_loss_func(
        cfg : Type[CONFIG],
        **kwargs
    ) -> nn.Module :

    if cfg.LOSS not in loss_dict :
        raise NotImplementedError("Invaild Loss Fucntion")
    
    return loss_dict[cfg.LOSS]( **kwargs )


def _get_score_func(
        cfg : CONFIG,
        **kwargs
    ) -> float :

    if cfg.SCORE not in score_dict :
        raise NotImplementedError("Invaild Score Function")
    
    return score_dict[cfg.SCORE]