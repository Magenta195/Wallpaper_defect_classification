import os
from typing import Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import CONFIG
from .optimizer import _get_optimizer
from .scheduler import _get_scheduler
from .metric import _get_loss_func, _get_score_func


class TRAINER():
    def __init__( 
        self, 
        model: nn.Module,
        dataloaders: DataLoader, 
        device: torch.device,
        cfg: Type[CONFIG],
        best_model_name: str = 'best_model',
    ) -> None:
        self.model = model
        self.trainloader = dataloaders['train']
        self.valloader = dataloaders['val']
        self.testloader = dataloaders['test']
        self.cfg = cfg
        self.device = device

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

        self.MODEL_SAVE_PATH = os.path.join(self.cfg.MODEL_SAVE_PATH, best_model_name + '.pth')
        self.cur_patience = 0
        self.softmax_for_predict = nn.Softmax()
        self.val_loss = float('inf')
        self.val_score = -float('inf')

        if cfg.METRIC_SCOPE == 'score':
            self.best_score = self.val_score
        else:
            self.best_score = self.val_loss

        self.model.to(self.device)

    def rand_bbox(size: Tuple[int], lam: float) -> Tuple[np.ndarray]: # size : [B, C, W, H]
        W = size[2] # 이미지의 width
        H = size[3] # 이미지의 height
        cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
        cut_w = np.int(W * cut_rat)  # 패치의 너비
        cut_h = np.int(H * cut_rat)  # 패치의 높이

        # uniform
        # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 패치 부분에 대한 좌표값을 추출합니다.
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    
    def _train_one_epoch(self) -> None:
        self.model.train()
        train_loss_list = []
        preds, true_labels = [], []
        self.train_loss = 0

        for idx, (imgs, labels) in enumerate(tqdm(self.trainloader)):
            beta = 1.0
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if beta > 0 and np.random.random()>0.5: # cutmix 작동될 확률      
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(imgs.size()[0]).to(self.device)
                target_a = labels
                target_b = labels[rand_index]            
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
                outputs = self.model(imgs)
                loss = self.criterion(outputs, target_a) * lam + self.criterion(outputs, target_b) * (1. - lam)
            else:
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels) 

            preds.append(outputs.argmax(1).data)                
            true_labels.append(labels.data)

            loss.backward()
            self.optimizer.step()
            if self.cfg.SCHEDULER == 'cosinewarmup':
                self.scheduler.step(self.cur_epoch + idx / len(self.trainloader))
            
            train_loss_list.append(loss.item())

        self.train_score = self.score_func(true_labels, preds, device=self.device, cfg=self.cfg, average = 'weighted')
        self.train_loss = np.mean(train_loss_list)

    def _val_one_epoch(self) -> None:
        self.model.eval()
        self.val_loss = 0
        val_loss_list = []
        preds, true_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(self.valloader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                pred = self.model(imgs)
                
                loss = self.criterion(pred, labels)

                preds.append(pred.argmax(1).data)                
                true_labels.append(labels.data)
                
                val_loss_list.append(loss.item())
        
        self.val_score = self.score_func(true_labels, preds, device=self.device, cfg=self.cfg, average='weighted')
        self.val_loss = np.mean(val_loss_list)

    def _is_best_model(self) -> bool:
        if self.cfg.METRIC_SCOPE == 'score':
            return self.val_score > self.best_score
        else:
            return self.val_loss < self.best_score
    
    def _cur_metric(self) -> float:
        if self.cfg.METRIC_SCOPE == 'score':
            return self.val_score
        else:
            return self.val_loss

    def _save_best_model(self) -> None:
        if self._is_best_model():
            torch.save(self.model, self.MODEL_SAVE_PATH)
            print("detected new best model, model save....")
            self.best_score = self._cur_metric()
            self.cur_patience = 0
        else :
            self.cur_patience += 1


    def _early_stopping(self) -> bool:
        if self.cfg.PATIENCE == 0:
            return False
        
        return self.cur_patience >= self.cfg.PATIENCE
        

    def full_train(self) -> None:
        self.train_loss_list = list()
        self.val_loss_list = list()

        for i in range(self.cfg.EPOCHS):
            self.cur_epoch = i
            self._train_one_epoch()
            self._val_one_epoch()
            self._save_best_model()
            if self.scheduler is not None and self.cfg.SCHEDULER != 'cosinewarmup':
                self.scheduler.step()

            print(" [ epoch : {:03d} ]  train_loss : {:0.03f}, train_score : {:0.03f}, val_loss : {:0.03f}, val_score : {:0.03f}, max_val_score : {:0.03f} ".format(
                i+1,
                self.train_loss,
                self.train_score,
                self.val_loss,
                self.val_score,
                self.best_score))
            if self._early_stopping():
                break


    def make_predict(self) -> np.ndarray:
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

    def make_predict_file(self) -> None:
        preds = self.make_predict()
        preds = np.argmax(preds, axis = 1)

        preds = [self.cfg.INV_CLASS_DICT[pred] for pred in preds]

        submit_df = pd.read_csv(self.cfg.SUBMIT_PATH)
        submit_df['label'] = preds
        submit_df.to_csv(self.cfg.OUTPUT_PATH, index=False)