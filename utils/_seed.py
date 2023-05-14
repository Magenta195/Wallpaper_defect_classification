import random
import os

import numpy as np
import torch


def set_seed(cfg):
    random.seed(cfg.SEED)
    os.environ['PYTHONHASHSEED'] = str(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True