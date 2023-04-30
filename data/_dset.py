import os
import glob

from torch.utils.data import Dataset
from utils import CONFIG




class CustomDataset(Dataset):
    def __init__(self, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def _get_data_list(self) :
        MODE_DATAPATH = os.path.join(CONFIG.DATA_PATH, 'train', '*', '*')
        train_ glob.glob(TRAIN_DATAPATH)
        
    def __len__(self):
        return len(self.img_path_list)