import os
from typing import Dict, Optional, Type

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
    CLASS_NUM = len(class_name)
    CLASS_DICT = {
        class_name[idx] : idx 
        for idx in range(len(class_name))
    }
    NUM_CLASSES = len(CLASS_DICT)
    INV_CLASS_DICT = {
        val : key
        for key, val in CLASS_DICT.items()
    }

    ## config for hyperparameters
    SEED = 1024
    KFOLD = 5
    EPOCHS = 20
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 8
    SEED = 41
    PATIENCE = 4

    ## config for model learning setting
    OPTIMIZER = 'Adam'
    OPTIMIZER_ARGS = dict()
    LOSS = 'celoss'
    LOSS_ARGS = dict()
    SCORE = 'f1score'
    SCHEDULER = None
    SCHEDULER_ARGS = dict()
    METRIC_SCOPE = 'score'

    ## ETC
    NUM_WORKER = 2


def config_init(
        cfg: Optional[Dict[str, any]] = None,
    ) ->  Type[CONFIG]:
    try :
        for key, val in cfg.items() :
            setattr(CONFIG, key, val)
    
    except :
        print("Set Default config setting")

    return CONFIG

if __name__ == '__main__' :
    cfg_dict = {
        'OPTIMIZER' : 'Adam',
        'OPTIMIZER_ARGS' : {
            'eps' : 1e-07,
            'weight_decay' : 0.1
        },
        'SCHEDULER' : 'cosine'
    }
    cfg = config_init(cfg_dict)
    print(cfg.OPTIMIZER)
    for key, val in cfg.OPTIMIZER_ARGS.items() :
        print(key, val)
    print(cfg.SCHEDULER)
    print(cfg.IMG_SIZE)
