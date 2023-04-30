
import os
from typing import Dict, Optional

class CONFIG :
    ## config for paths
    DATA_PATH = './data'
    MODEL_SAVE_PATH = './best_model.pth'
    OUTPUT_PATH = './submission.csv'
    SUBMIT_PATH = os.path.join(DATA_PATH, 'sample_submission.csv')

    ## config for hyperparameters
    IMG_SIZE = 224
    EPOCHS = 20
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    SEED = 41
    PATIENCE = 4

def config_init(
        cfg: Optional[Dict]
    ) ->  None:
    try :
        for key, val in cfg.items() :
            setattr(CONFIG, key, val)
    
    except :
        print("Set Default config setting")
