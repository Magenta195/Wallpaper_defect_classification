from typing import Type

import torch.nn as nn
import torch.optim

from utils import CONFIG


OPTIMIZER_DICT = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
    'Adamw': torch.optim.AdamW,
    'Adamax': torch.optim.Adamax,
}


def _get_optimizer(
    model_param: any,
    cfg: Type[CONFIG],
    **kwargs
) -> nn.Module:
    if cfg.OPTIMIZER not in OPTIMIZER_DICT:
        raise NotImplementedError("Invaild Optimizer")

    return OPTIMIZER_DICT[cfg.OPTIMIZER](
        params=model_param,
        lr=cfg.LEARNING_RATE,
        **cfg.OPTIMIZER_ARGS
    )
