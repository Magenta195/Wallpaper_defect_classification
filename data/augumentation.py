from typing import Type

import numpy as np
import torch.nn as nn
from torchvision import transforms

from utils import CONFIG


def train_transforms(
        cfg: Type[CONFIG]
    ) -> nn.Sequential:
    """This function is transforms for train dataset"""
    return transforms.Compose([
        transforms.RandomResizedCrop((cfg.IMG_SIZE, cfg.IMG_SIZE), scale=(0.8, 1.0)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def test_transforms(
        cfg: Type[CONFIG]
    ) -> nn.Sequential:
    """This function is transforms for test dataset"""
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    

