
from typing import Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from utils import CONFIG

class TRAINER() :
    def __init__( self, 
        model : nn.Module,
        trainloader : DataLoader,
        valloader : DataLoader,
        testloader : DataLoader, 
        device : torch.device,
        scheduler : Optional[any],     
    ):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.scheduler = scheduler
        self.device = device

        self.cur_patience = 0

        self.cur_score = -float('inf')
        self.best_score = -float('inf')


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
            
            self._train_loss.append(loss.item())

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
                
                self._val_loss.append(loss.item())
            
        self.val_loss = np.mean(val_loss_list)

    def _save_best_model( self ) :
        if self.cur_score < self.best_score :
            torch.save(self.model, CONFIG.MODEL_SAVE_PATH)
            print("detected new best model, model save....")
            self.cur_patience = 0
        else :
            self.cur_patience += 1

    def _early_stopping( self ) :
        if CONFIG.PATIENCE == 0 :
            return False
        
        return self.cur_patience >= CONFIG.PATIENCE
        

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

        self.model.eval()
        preds = []
        with torch.no_grad():
            for imgs in tqdm(self.testloader):
                imgs = imgs.to(self.device)
                
                pred = self.model(imgs)
                
                preds += pred.argmax(1).detach().cpu().numpy().tolist()

        submit_df = pd.read_csv(CONFIG.SUBMIT_PATH)
        submit_df['label'] = preds
        submit_df.to_csv(CONFIG.OUTPUT_PATH, index=False)

    