from torch import Tensor
import torch.nn as nn

class CELoss(nn.CrossEntropyLoss) :
    def __init__(self, **kwargs) :
        super().__init__(self, **kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target)