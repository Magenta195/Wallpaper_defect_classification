from typing import List, Dict, Union, Tuple, Optional, Type
import unicodedata

from torch.utils.data import DataLoader, random_split, Subset, Dataset

from .utils import CONFIG
from ._argumentation import *
from ._dset import *

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