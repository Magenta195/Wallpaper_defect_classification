from typing import Dict, Type

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

from utils import CONFIG
from .dset import *
from .augumentation import *
from .dutils import *


__all__ = [WallPaperDataset]

MODE_LIST = ['train', 'val', 'test']


def get_all_dataloader(
        cfg: Type[CONFIG],
        val_size: float = 0.1,
        image_folder: bool = False
    ) -> Dict[str, DataLoader]:
    """This function get the dict consist of train, validation, test Dataloader"""
    dataloader_dict = dict()
    datalist_dict = dict()
    sampler_dict = {mode: None for mode in MODE_LIST}
    # Load image dataset using torchvision.datasets.ImageFolder
    if image_folder:    
        for mode in ['train', 'test']:
            # Get ImageFolder
            img_folder = get_image_folder(mode, cfg)
            # Stratified split train, validation set
            if mode == 'train' :
                targets = img_folder.targets
                train_idx, valid_idx = train_test_split(
                    np.arange(len(targets)),
                    test_size=val_size,
                    stratify=targets,
                    random_state = cfg.SEED,
                )
                sampler_dict['train'] = SubsetRandomSampler(train_idx)
                sampler_dict['val'] = SubsetRandomSampler(valid_idx)
                datalist_dict['val'] = img_folder

            datalist_dict[mode] = img_folder
    # Load image dataset using data path 
    else:
        for mode in ['train', 'test']:
            # Get image data path list
            img_list, label_list = get_data_list( mode = mode, cfg = cfg )
            # Split train, validation set
            if mode == 'train':
                train_img_list, val_img_list, train_label_list, val_label_list = train_test_split(
                    img_list, label_list, 
                    test_size = val_size,
                    random_state = cfg.SEED,
                    stratify = label_list
                )
                datalist_dict['train'] = [train_img_list, train_label_list]
                datalist_dict['val'] = [val_img_list, val_label_list]
            else :
                datalist_dict[mode] = [img_list, label_list]

    # Make the dataset to DataLoader
    for mode in MODE_LIST:
        if image_folder:
            img_list, label_list = None, None
            img_folder = datalist_dict[mode]
        else:
            img_list, label_list = datalist_dict[mode]
            img_folder = None

        dataloader_dict[mode] = get_dataloader(
            mode = mode,
            cfg = cfg,
            img_path_list = img_list,
            label_list = label_list,
            dataset = img_folder,
            sampler = sampler_dict[mode]
        )
    
    return dataloader_dict


__all__.extend([get_dataloader, get_data_list, get_all_dataloader, train_transforms, test_transforms])