
from typing import Type
import numpy as np

import torch.nn as nn
from torchvision import transforms
from utils import CONFIG

def train_transforms(
        cfg : Type[CONFIG],
) -> nn.Sequential :
    return transforms.Compose(
    [transforms.RandomResizedCrop((cfg.IMG_SIZE, cfg.IMG_SIZE), scale=(0.8, 1.0)),
     transforms.RandomRotation(30),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def test_transforms(
        cfg : Type[CONFIG],
) -> nn.Sequential :
    return transforms.Compose(
    [transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
def rand_bbox(size, lam): # size : [B, C, W, H]
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