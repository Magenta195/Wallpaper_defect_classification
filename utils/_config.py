
import os
from typing import Dict, Optional

class_name = [
    '가구수정', '걸레받이수정', '곰팡이', '꼬임', '녹오염', '들뜸', '면불량',
    '몰딩수정', '반점', '석고수정', '오염', '오타공', '울음', '이음부불량',
    '창틀,문틀수정', '터짐', '틈새과다', '피스', '훼손'
]

class CONFIG :
    ## config for paths
    DATA_PATH = './data'
    MODEL_SAVE_PATH = './'
    OUTPUT_PATH = './submission.csv'
    SUBMIT_PATH = os.path.join(DATA_PATH, 'sample_submission.csv')

    ## config for data argumentation
    IMG_SIZE = 224
    CLASS_DICT = {
        class_name[idx] : idx 
        for idx in range(len(class_name))
    }
    INV_CLASS_DICT = {
        val : key
        for key, val in CLASS_DICT.items()
    }

    ## config for hyperparameters
    
    EPOCHS = 20
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    SEED = 41
    PATIENCE = 4

    OPTIMIZER = 'Adam'
    LOSS = 'celoss'
    SCORE = 'f1score'
    SCHEDULER = None

    ## ETC

    NUM_WORKER = 2


def config_init(
        cfg: Optional[Dict]
    ) ->  CONFIG:
    try :
        for key, val in cfg.items() :
            setattr(CONFIG, key, val)
    
    except :
        print("Set Default config setting")

    return CONFIG
