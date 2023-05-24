from typing import List, Tuple, Optional, Union

import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import PIL
from PIL import Image


class WallPaperDataset(Dataset):
    def __init__(
            self, 
            img_path_list: List[str], 
            label_list: List[int], 
            transforms: Optional[nn.Module] = None
        ) -> None:
        self.label_list = label_list
        self.transforms = transforms
        self.img_list = list()
        for img_path in tqdm(img_path_list):
            image = Image.open(img_path)
            self.img_list.append(image)
        
    def __getitem__(self, index) -> Union[Tuple[PIL.Image, int], Tuple[PIL.Image, None]]:
        image = self.img_list[index]

        if self.transforms is not None:
            image = self.transforms(image)
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image, None
        
    def __len__(self) -> int:
        return len(self.img_list)