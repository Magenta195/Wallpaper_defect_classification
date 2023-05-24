from typing import Type, List

import torch
from torchmetrics.classification import MulticlassF1Score

from utils import CONFIG, config_init


def f1_score(
        labels: List[torch.Tensor],
        preds: List[torch.Tensor],
        device: torch.device,
        cfg: Type[CONFIG],
        average: str = 'weighted',
        **kwargs
    ) -> float:
    """This function get F1 score"""
    metric = MulticlassF1Score(cfg.NUM_CLASSES, average=average)
    metric.to(device)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    score = metric(preds, labels)

    return score.item()


if __name__ == "__main__":
    import torch.nn as nn


    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cur device: ', device)

    # config setting
    config_init({'NUM_CLASSES': 3})
    cfg = CONFIG
    print('cfg num_classes: ', cfg.NUM_CLASSES)

    # make test preds, labels
    preds, labels = [], []
    m = nn.Linear(20, 3)
    m.to(device)
    for _ in range(2):
        label = torch.randint(0, 3, (128,))
        label = label.to(device)
        labels.append(label.data)
        input = torch.randn(128, 20)
        input = input.to(device)
        output = m(input)
        output = output.argmax(1)
        preds.append(output.data)

    s = f1_score(labels, preds, device, cfg=cfg)

    # print result
    print(s)
    print(type(s))