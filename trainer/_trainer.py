
import os
from typing import Optional, Type
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from utils import CONFIG
from ._optimizer import _get_optimizer
from ._scheduler import _get_scheduler
from ._metric import _get_loss_func, _get_score_func

class TRAINER() :
    def __init__( self, 
        model : nn.Module,
        dataloaders : DataLoader, 
        device : torch.device,
        cfg : Type[CONFIG],
        best_model_name : str = 'best_model',
    ):
        self.model = model
        self.trainloader = dataloaders['train']
        self.valloader = dataloaders['val']
        self.testloader = dataloaders['test']
        self.cfg = cfg

        self.optimizer = _get_optimizer(
            model_param = self.model.parameters(),
            cfg = cfg
        )
        self.scheduler = _get_scheduler(
            optimizer = self.optimizer,
            cfg = cfg
        )
        self.criterion = _get_loss_func(
            cfg = cfg
        )
        self.score_func = _get_score_func(
            cfg = cfg
        )
        self.device = device

        self.MODEL_SAVE_PATH = os.path.join(self.cfg.MODEL_SAVE_PATH, best_model_name + '.pth')
        self.cur_patience = 0
        self.softmax_for_predict = nn.Softmax()

        self.cur_score = -float('inf')
        self.best_score = -float('inf')

        self.model.to(self.device)


    def _train_one_epoch( self ) -> None :
        self.model.train()
        train_loss_list = []
        self.train_loss = 0

        for imgs, labels in tqdm(self.trainloader):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(imgs)
            loss = self.criterion(output, labels)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss_list.append(loss.item())

        self.train_loss = np.mean(train_loss_list)


    def _val_one_epoch( self ) -> None :
        self.model.eval()
        val_loss_list = []
        preds, true_labels = [], []
        self.val_loss = 0

        with torch.no_grad():
            for imgs, labels in tqdm(self.valloader) :
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                pred = self.model(imgs)
                
                loss = self.criterion(pred, labels)

                preds.append(pred.argmax(1).data)                
                true_labels.append(labels.data)
                
                val_loss_list.append(loss.item())
        
        self.cur_score = self.score_func(true_labels, preds, device=self.device, cfg=self.cfg, average = 'weighted')
        self.val_loss = np.mean(val_loss_list)


    def _save_best_model( self ) -> None :
        if self.cur_score > self.best_score :
            torch.save(self.model, self.MODEL_SAVE_PATH)
            print("detected new best model, model save....")
            self.best_score = self.cur_score
            self.cur_patience = 0
        else :
            self.cur_patience += 1


    def _early_stopping( self ) -> bool :
        if self.cfg.PATIENCE == 0 :
            return False
        
        return self.cur_patience >= self.cfg.PATIENCE
        

    def full_train( self ) -> None :
        self.train_loss_list = list()
        self.val_loss_list = list()

        for i in range( 1, self.cfg.EPOCHS+1 ) :
            self._train_one_epoch()
            self._val_one_epoch()
            self._save_best_model()
            if self._early_stopping() :
                break

            if self.scheduler is not None :
                self.scheduler.step()

            print( " [ epoch : {:03d} ] train_loss : {:0.03f}, val_loss : {:0.03f}, val_score : {:0.03f}, max_val_score : {:0.03f} ".format(
                i,
                self.train_loss,
                self.val_loss,
                self.cur_score,
                self.best_score))


    def make_predict( self ) -> np.ndarray :
        self.model = torch.load(self.MODEL_SAVE_PATH)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for imgs, _ in tqdm(self.testloader):
                imgs = imgs.to(self.device)
                
                pred = self.softmax_for_predict(self.model(imgs))
                
                preds.append(pred.detach().cpu().numpy())
        preds = np.concatenate(preds, axis = 0)
        return preds


    def make_predict_file( self ) -> None :
        preds = self.make_predict()
        preds = np.argmax(preds, axis = 1)

        preds = [ self.cfg.INV_CLASS_DICT[pred] for pred in preds ]

        submit_df = pd.read_csv( self.cfg.SUBMIT_PATH )
        submit_df['label'] = preds
        submit_df.to_csv( self.cfg.OUTPUT_PATH, index=False )
    