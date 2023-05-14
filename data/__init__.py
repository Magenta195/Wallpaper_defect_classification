from typing import List, Dict, Union, Tuple, Optional, Type

from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler
import numpy as np

from utils import CONFIG
from ._dset import *
from ._argumentation import *
from ._dutils import *

__all__ = [WallPaperDataset]

mode_list = ['train', 'val', 'test']

def get_all_dataloader(
        cfg : Type[CONFIG],
        val_size : float = 0.1,
        image_folder : bool = False,
    ) -> Dict[str, DataLoader] :
    """This function get the dict consist of train, validation, test Dataloader"""
    dataloader_dict = dict()
    datalist_dict = dict()
    sampler_dict = {mode: None for mode in mode_list}
    # using torchvision.datasets.ImageFolder
    if image_folder:    
        for mode in ['train', 'test'] :
            img_folder = get_image_folder(mode, cfg)
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
    else:
        for mode in ['train', 'test'] :
            img_list, label_list = get_data_list( mode = mode, cfg = cfg )
            if mode == 'train' :
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

    for mode in mode_list :
        if image_folder :
            img_list, label_list = None, None
            img_folder = datalist_dict[mode]
        else :
            img_list, label_list = datalist_dict[mode]
            img_folder = None

        dataloader_dict[mode] = get_dataloader(
            mode = mode,
            cfg = cfg,
            img__path_list = img_list,
            label_list = label_list,
            dataset = img_folder,
            sampler = sampler_dict[mode]
        )
    
    return dataloader_dict

def get_kfold_dataloader(
        cfg : Type[CONFIG],
    ) -> Dict[str, DataLoader] :

    dataloader_dict = dict()
    datalist_dict = dict()
    kfold = StratifiedKFold(n_splits = cfg.KFOLD, random_state = cfg.SEED)
    # Only use torchvision.datasets.ImageFolder
  
    for mode in ['train', 'test'] :
        img_folder = get_image_folder(mode, cfg)
        if mode == 'train' :
            label_list =  [ x for _, x in img_folder.samples ]
            for i, (train_idx, val_idx) in enumerate(kfold.split(img_folder, label_list)) :
                train_dset = Subset(img_folder, train_idx)
                val_dset = Subset(img_folder, val_idx)
                val_dset.transform = test_transforms( cfg = cfg )
                datalist_dict[str(i)] =  ( train_dset, val_dset )
        else:
            datalist_dict['test'] = ( img_folder, None )
 
    for mode, (dset1, dset2) in datalist_dict.items() :
        if mode == 'test' :
            dataloader_dict[mode] = get_dataloader(
                        mode = 'test',
                        cfg = cfg,
                        img_path_list = None,
                        label_list = None,
                        dataset = dset1
                    )
        else :
            dataloader_dict['train' + mode] = get_dataloader(
                        mode = 'train',
                        cfg = cfg,
                        img_path_list = None,
                        label_list = None,
                        dataset = dset1
                    )
            dataloader_dict['val' + mode] = get_dataloader(
                        mode = 'val',
                        cfg = cfg,
                        img_path_list = None,
                        label_list = None,
                        dataset = dset2
                    )
                            
    return dataloader_dict


__all__.extend([get_dataloader, get_data_list, get_all_dataloader, get_kfold_dataloader, train_transforms, test_transforms])