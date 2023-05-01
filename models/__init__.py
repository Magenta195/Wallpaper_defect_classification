import torch.nn as nn

from ._effinet import _get_effinet_models

model_dict = {
    'effinet' : _get_effinet_models,
}

def get_torch_model(
        model_name : str,
        **kwargs
    ) -> nn.Module :

    if model_name not in model_dict :
        print("detected unknown model name...")
        print("List of models can be used currently") 
        print("--------------")
        for idx, key in enumerate(model_dict.keys()) :
            print( '[{:03d}] {}'.format(idx+1, key) )

        # raise NotImplementedError
        return None
    
    for key, val in kwargs.items() :
        print(key, val)

    return model_dict[ model_name ]( **kwargs )

__all__ = [get_torch_model]