
from typing import Dict, Optional
from abc import *


class CONFIG :
    IMG_SIZE = 224
    EPOCHS = 10
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    SEED = 41

def config_init(
        cfg: Optional[Dict]
    ) ->  None:
    try :
        for key, val in cfg.items() :
            setattr(CONFIG, key, val)
    
    except :
        print("Set Default config setting")
