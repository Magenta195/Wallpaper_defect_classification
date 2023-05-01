from typing import List, Dict, Union, Tuple, Optional
import glob
import os 

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .utils import CONFIG
from ._dset import *
from ._argumentation import *

__all__ = [WallPaperDataset]

mode_list = ['train', 'val', 'test']

def get_dataloader(
        mode : str,
        cfg : CONFIG,
        img_path_list : List[str],
        label_list : Optional[List[str]]
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
        cfg : CONFIG,
    ) -> Union[Tuple[List[str], None], Tuple[List[str], List[str]]] :
    
    if mode == 'train' :
        TRAIN_DATA_PATH = os.path.join(cfg.DATA_PATH, 'train', '*', '*')
        img_list = glob.glob(TRAIN_DATA_PATH)
        label_list  = list()

        for img_dir in img_list :
            label_name = str(img_dir).split('/')[-2]
            label_list.append(cfg.CLASS_DICT[label_name])

        return img_list, label_list

    else :
        TEST_DATA_PATH = os.path.join(cfg.DATA_PATH, 'test', '*')
        img_list = glob.glob(TEST_DATA_PATH)

        return img_list, None
    
def get_all_dataloader(
        cfg : CONFIG,
        val_size : float = 0.1,
    ) -> Dict[str, DataLoader] :
    dataloader_dict = dict()
    datalist_dict = dict()
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

__all__.extend([get_dataloader, get_data_list, get_all_dataloader])