import torch.nn as nn

from .effinet import _get_effinet_models
from .effinetv2 import _get_effinetv2_models


MODEL_DICT = {
    'effinet': _get_effinet_models,
    'effinetv2': _get_effinetv2_models,
}


def get_torch_model(
        model_name: str,
        **kwargs
    ) -> nn.Module:
    """This function return torch.nn.Module"""
    if model_name not in MODEL_DICT:
        print("detected unknown model name...")
        print("List of models can be used currently") 
        print("--------------")
        for idx, key in enumerate(MODEL_DICT.keys()):
            print('[{:03d}] {}'.format(idx+1, key))

        # raise NotImplementedError
        return None
    
    for key, val in kwargs.items():
        print(key, val)

    return MODEL_DICT[model_name](**kwargs)


__all__ = [get_torch_model]