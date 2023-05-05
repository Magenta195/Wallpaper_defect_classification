from typing import List, Dict, Union, Tuple, Optional, Type
import glob
import os 
import unicodedata

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from utils import CONFIG
from ._dset import *
from ._argumentation import *

__all__ = [WallPaperDataset]

mode_list = ['train', 'val', 'test']

def get_dataloader(
        mode : str,
        cfg : Type[CONFIG],
        img_path_list : List[str],
        label_list : Optional[List[str]],
    ) -> DataLoader :
    
    if mode not in mode_list :
        raise ValueError("Invaild DataLoader Type")
    
    if mode == 'train' :
        transform = train_transforms( cfg = cfg )
        shuffle = True
    else :
        transform = test_transforms( cfg = cfg )
        shuffle = False

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
                train_folder, val_folder = random_split(img_folder, [0.9, 0.1])
                datalist_dict['train'] = train_folder
                datalist_dict['val'] = val_folder
            else:
                datalist_dict['test'] = img_folder
            
        for mode in mode_list:
            img_folder = datalist_dict[mode]
            dataloader_dict[mode] = DataLoader(
                                        dataset = img_folder,
                                        batch_size = cfg.BATCH_SIZE,
                                        shuffle = True if mode in ['train', 'val'] else False,
                                        num_workers = cfg.NUM_WORKER
                                    )
    else:
        for mode in ['train', 'test'] :
            img_list, label_list = get_data_list( mode = mode, cfg = cfg )

            if mode == 'train' :
                train_img_list, val_img_list, train_label_list, val_label_list = train_test_split(
                    img_list, label_list, 
                    test_size = val_size,
                    train_size = 1 - val_size,
                )

                datalist_dict['train'] = [train_img_list, train_label_list]
                datalist_dict['val'] = [val_img_list, val_label_list]

            else :
                datalist_dict[mode] = [img_list, label_list]

        for mode in mode_list :
            img_list, label_list = datalist_dict[mode]
            dataloader_dict[mode] = get_dataloader(
                        mode = mode,
                        cfg = cfg,
                        img_path_list = img_list,
                        label_list = label_list)
    
    return dataloader_dict

__all__.extend([get_dataloader, get_data_list, get_all_dataloader, train_transforms, test_transforms])