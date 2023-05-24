from typing import Type

import torch.nn as nn
from utils import CONFIG
from .loss import CELoss
from .score import f1_score


LOSS_DICT = {
    'celoss': CELoss,
}

SCORE_DICT = {
    'f1score': f1_score
}


def _get_loss_func(
        cfg: Type[CONFIG],
        **kwargs
    ) -> nn.Module:
    if cfg.LOSS not in LOSS_DICT:
        raise NotImplementedError("Invaild Loss Fucntion")
    
    return LOSS_DICT[cfg.LOSS](**cfg.LOSS_ARGS)


def _get_score_func(
        cfg: CONFIG,
        **kwargs
    ) -> float:
    if cfg.SCORE not in SCORE_DICT:
        raise NotImplementedError("Invaild Score Function")
    
    return SCORE_DICT[cfg.SCORE]