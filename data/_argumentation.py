
import torch.nn as nn
from torchvision import transforms
from utils import CONFIG
    

def train_transforms() -> nn.Sequential :
    return transforms.Compose(
    [transforms.RandomResizedCrop((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), scale=(0.8, 1.0)),
     #transforms.Grayscale(num_output_channels=3),
     transforms.RandomRotation(30),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
     #transforms.Normalize((0.485), (0.229))
    ])

def test_transforms() -> nn.Sequential :
    return transforms.Compose(
    [transforms.Resize((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE)),
     #transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
     #transforms.Normalize((0.485), (0.229))
    ])