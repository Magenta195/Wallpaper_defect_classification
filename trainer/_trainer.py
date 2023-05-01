
import os
from typing import Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from .utils import CONFIG
from ._optimizer import _get_optimizer
from ._scheduler import _get_scheduler
from ._metric import _get_loss_func, _get_score_func

class TRAINER() :
    def __init__( self, 
        model : nn.Module,
        dataloaders : DataLoader, 
        device : torch.device,
        cfg : CONFIG,
        best_model_name : str = 'best_model',
    ):
        self.model = model
        self.trainloader = dataloaders['train']
        self.valloader = dataloaders['val']
        self.testloader = dataloaders['test']
        self.cfg = cfg

        self.optimizer = _get_optimizer(
            opt_name = cfg.OPTIMIZER,
            model_param = self.model,
            cfg = cfg
        )
        self.scheduler = _get_scheduler(
            scheduler_name = cfg.SCHEDULER,
            optimizer = self.optimizer,
            cfg = cfg
        )
        self.criterion = _get_loss_func(
            loss_name = cfg.LOSS,
            cfg = cfg
        )
        self.score_func = _get_score_func(
            score_name = cfg.SCORE,
            cfg = cfg
        )
        self.device = device

        self.MODEL_SAVE_PATH = os.path.join(CONFIG.MODEL_SAVE_PATH, best_model_name + '.pth')
        self.cur_patience = 0
        self.softmax_for_predict = nn.Softmax()

        self.cur_score = -float('inf')
        self.best_score = -float('inf')

        self.model.to(self.device)
        self.optimizer.to(self.device)

    def _train_one_epoch( self ) :
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

    def _val_one_epoch( self ) :
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
                
                preds += pred.argmax(1).detach().cpu().numpy().tolist()
                true_labels += labels.detach().cpu().numpy().tolist()
                
                val_loss_list.append(loss.item())
        
        self.cur_score = self.score_func(true_labels, preds, average = 'weighted')
        self.val_loss = np.mean(val_loss_list)

    def _save_best_model( self ) :
        if self.cur_score < self.best_score :
            torch.save(self.model, self.MODEL_SAVE_PATH)
            print("detected new best model, model save....")
            self.cur_patience = 0
        else :
            self.cur_patience += 1

    def _early_stopping( self ) :
        if self.cfg.PATIENCE == 0 :
            return False
        
        return self.cur_patience >= self.cfg.PATIENCE
        

    def full_train( self ) :
        self.train_loss_list = list()
        self.val_loss_list = list()

        for i in range( 1, CONFIG.EPOCHS+1 ) :
            self._train_one_epoch()
            self._val_one_epoch()
            self._save_best_model()
            if self._early_stopping() :
                break

        if self.scheduler is not None :
            self.scheduler.step()

    def make_predict( self ) :
        self.model = torch.load(self.MODEL_SAVE_PATH)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for imgs in tqdm(self.testloader):
                imgs = imgs.to(self.device)
                
                pred = self.softmax_for_predict(self.model(imgs))
                
                preds.append(pred.detach().cpu().numpy())
        preds = np.concatenate(preds, axis = 0)
        return preds

    def make_predict_file( self ) :
        preds = self.make_predict()
        preds = np.argmax(preds, axis = 1)

        preds = [ self.cfg.INV_CLASS_DICT[pred] for pred in preds ]

        submit_df = pd.read_csv( self.cfg.SUBMIT_PATH )
        submit_df['label'] = preds
        submit_df.to_csv( self.cfg.OUTPUT_PATH, index=False )
    