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
        loss_name : str,
        cfg : CONFIG,
        **kwargs
    ) -> nn.Module :

    if loss_name not in loss_dict :
        raise NotImplementedError("Invaild Loss Fucntion")
    
    return loss_dict[loss_name]( **kwargs )


def _get_score_func(
        score_name : str,
        cfg : CONFIG,
        **kwargs
    ) -> float :

    if score_name not in score_dict :
        raise NotImplementedError("Invaild Score Function")
    
    return score_dict[score_name]