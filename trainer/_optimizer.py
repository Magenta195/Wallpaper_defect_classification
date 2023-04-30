
import torch.nn as nn
import torch.optim

from utils import CONFIG

optimizer_dict = {
    'Adam' : torch.optim.Adam,
    'SGD' : torch.optim.SGD,
    'Adamw' : torch.optim.AdamW
}

def _get_optimizer(
        opt_name : str,
        model_param : any,
        **kwargs
    ) -> nn.Module :

    if opt_name not in optimizer_dict :
        raise NotImplementedError("Invaild Optimizer")
    
    return optimizer_dict[opt_name]( params = model_param,
                                    lr = CONFIG.LEARNING_RATE,
                                    **kwargs )
