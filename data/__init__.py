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
        img_path_list : List[str],
        label_list : Optional[List[str]]
    ) -> DataLoader :
    
    if mode not in mode_list :
        raise ValueError("Invaild DataLoader Type")
    
    if mode == 'train' :
        transform = train_transforms()
        shuffle = True
    else :
        transform = test_transforms()
        shuffle = False

    dataset = WallPaperDataset(
        img_path_list = img_path_list,
        label_list = label_list,
        transforms = transform
    )

    return DataLoader(
        dataset = dataset,
        batch_size = CONFIG.BATCH_SIZE,
        shuffle = shuffle,
        num_workers = CONFIG.NUM_WORKER
    )

def get_data_list(
        mode : str
    ) -> Union[Tuple[List[str], None], Tuple[List[str], List[str]]] :
    
    if mode == 'train' :
        TRAIN_DATA_PATH = os.path.join(CONFIG.DATA_PATH, 'train', '*', '*')
        img_list = glob.glob(TRAIN_DATA_PATH)
        label_list  = list()

        for img_dir in img_list :
            label_name = str(img_dir).split('/')[-2]
            label_list.append(CONFIG.CLASS_DICT[label_name])

        return img_list, label_list

    else :
        TEST_DATA_PATH = os.path.join(CONFIG.DATA_PATH, 'test', '*')
        img_list = glob.glob(TEST_DATA_PATH)

        return img_list, None
    
def get_all_dataloader(
        val_size : float = 0.1
    ) -> Dict[DataLoader] :
    dataloader_dict = dict()
    datalist_dict = dict()
    for mode in ['train', 'test'] :
        img_list, label_list = get_data_list( mode = mode )

        if mode == 'train' :
            train_img_list, val_img_list, train_label_list, val_label_list = train_test_split(
                img_list, label_list, test_size = val_size
            )

            datalist_dict['train'] = [train_img_list, train_label_list]
            datalist_dict['val'] = [val_img_list, val_label_list]

        else :
            datalist_dict[mode] = [img_list, label_list]


    for mode in mode_list :
        img_list, label_list = datalist_dict[mode]
        dataloader_dict[mode] = get_dataloader(mode = mode,
                    img_path_list = img_list,
                    label_list = label_list)
    
    return dataloader_dict

__all__.extend([get_dataloader, get_data_list, get_all_dataloader])