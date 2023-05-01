from typing import Optional, List, Type

import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from .utils import CONFIG
from ._trainer import TRAINER

class ENSEMBLE() : ### just using voting mechanism
    def __init__( self, 
        model_list : List[nn.Module],
        dataloaders : DataLoader,
        device : torch.device,
        cfg : Type[CONFIG],
        best_model_name : str = 'best_model',
        scheduler : Optional[str] = None, 
    ):
        self.cfg = cfg
        self.trainer_list = [
                TRAINER(
                    model = _model,
                    dataloaders = dataloaders,
                    device = device,
                    cfg = cfg,
                    best_model_name = best_model_name + str(idx),
                    scheduler = scheduler
            ) for idx, _model in enumerate(model_list)
        ]
        
    def train_all( self ) :
        for idx, trainer in enumerate(self.trainer_list) :
            print('[Model {:03d} training start]'.format(idx + 1))
            trainer.full_train()

    def make_predict( 
            self,
            mode : str = 'soft'
        ) -> np.ndarray :
        preds = None
        
        for trainer in self.trainer_list :
            result = trainer.make_predict()
            if mode == 'hard' :
                result = np.where(result == np.max(result, axis = 1, keepdims=True), 1, 0)

            if preds is None :
                preds = result
            else :
                preds += result

        return preds
    
    def make_predict_file(
            self,
            mode : str = 'soft'
        ) -> None :

        preds = self.make_predict( mode )
        preds = np.argmax(preds, axis = 1)

        preds = [ self.cfg.INV_CLASS_DICT[pred] for pred in preds ]

        submit_df = pd.read_csv( self.cfg.SUBMIT_PATH )
        submit_df['label'] = preds
        submit_df.to_csv( self.cfg.OUTPUT_PATH, index=False )