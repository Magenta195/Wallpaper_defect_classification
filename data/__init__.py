from typing import List, Dict, Union, Tuple, Optional, Type
import glob
import os 
import unicodedata

from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from .utils import CONFIG
from ._dset import *
from ._argumentation import *

__all__ = [WallPaperDataset]

mode_list = ['train', 'val', 'test']

def get_dataloader(
        mode : str,
        cfg : Type[CONFIG],
        img_path_list : Optional[List[str]] = None,
        label_list : Optional[List[str]] = None,
        dataset : Optional[Dataset] = None,
    ) -> DataLoader :
    
    if mode not in mode_list :
        raise ValueError("Invaild DataLoader Type")
    
    if mode == 'train' :
        transform = train_transforms( cfg = cfg )
        shuffle = True
    else :
        transform = test_transforms( cfg = cfg )
        shuffle = False

    if dataset is None :
        dataset = WallPaperDataset(
            img_path_list = img_path_list,
            label_list = label_list,
            transforms = transform
        )

    return DataLoader(
        dataset = dataset,
        batch_size = cfg.BATCH_SIZE,
        shuffle = shuffle,
        num_workers = cfg.NUM_WORKER
    )

def get_data_list(
        mode : str,
        cfg : Type[CONFIG],
    ) -> Union[Tuple[List[str], None], Tuple[List[str], List[str]]] :
    
    if mode == 'train' :
        TRAIN_DATA_PATH = os.path.join(cfg.DATA_PATH, 'train', '*', '*')
        img_list = glob.glob(TRAIN_DATA_PATH)
        label_list  = list()

        for img_dir in img_list :
            label_name = str(img_dir).split('/')[-2]
            try :
                label = cfg.CLASS_DICT[label_name]
            except :
                label = cfg.CLASS_DICT[unicodedata.normalize('NFC', label_name)]
            label_list.append(label)

        return img_list, label_list

    else :
        TEST_DATA_PATH = os.path.join(cfg.DATA_PATH, 'test', 'test', '*')
        img_list = glob.glob(TEST_DATA_PATH)

        return img_list, None
    

def get_image_folder(
    mode : str,
    cfg : Type[CONFIG],
) -> ImageFolder:
    DATA_PATH = os.path.join(cfg.DATA_PATH, mode)
    if mode == 'train':
        img_folder = ImageFolder(root=DATA_PATH, transform=train_transforms( cfg = cfg ))   
    else:
        img_folder = ImageFolder(root=DATA_PATH, transform=test_transforms( cfg = cfg ))

    return img_folder

def get_all_dataloader(
        cfg : Type[CONFIG],
        val_size : float = 0.1,
        image_folder : bool = False,
    ) -> Dict[str, DataLoader] :

    dataloader_dict = dict()
    datalist_dict = dict()
    # using torchvision.datasets.ImageFolder
    if image_folder:    
        for mode in ['train', 'test'] :
            img_folder = get_image_folder(mode, cfg)
            if mode == 'train' :
                train_folder, val_folder = random_split(img_folder, [1-val_size, val_size])
                datalist_dict['train'] = train_folder
                datalist_dict['val'] = val_folder
            else:
                datalist_dict['test'] = img_folder
            
    else:
        for mode in ['train', 'test'] :
            img_list, label_list = get_data_list( mode = mode, cfg = cfg )
            if mode == 'train' :
                train_img_list, val_img_list, train_label_list, val_label_list = train_test_split(
                    img_list, label_list, 
                    test_size = val_size,
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
                    img_path_list = img_list,
                    label_list = label_list,
                    dataset = img_folder
                )
    
    return dataloader_dict

def get_kfold_dataloader(
        cfg : Type[CONFIG],
    ) -> Dict[str, DataLoader] :

    dataloader_dict = dict()
    datalist_dict = dict()

    kfold = StratifiedKFold(n_splits = cfg.KFOLD)
    # Only use torchvision.datasets.ImageFolder
  
    for mode in ['train', 'test'] :
        img_folder = get_image_folder(mode, cfg)
        if mode == 'train' :
            label_list =  [ x for _, x in img_folder.samples ]
            for i, (train_dset, val_dset) in enumerate(kfold.split(img_folder, label_list)) :
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