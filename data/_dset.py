from PIL import Image

from torch.utils.data import Dataset

# class WallPaperDataset(Dataset):
#     def __init__(self, img_path_list, label_list, transforms=None):
#         self.img_path_list = img_path_list
#         self.label_list = label_list
#         self.transforms = transforms
        
#     def __getitem__(self, index):
#         img_path = self.img_path_list[index]
        
#         image = Image.open(img_path)
        
#         if self.transforms is not None:
#             image = self.transforms(image)
        
#         if self.label_list is not None:
#             label = self.label_list[index]
#             return image, label
#         else:
#             return image
        
#     def __len__(self):
#         return len(self.img_path_list)

class WallPaperDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.label_list = label_list
        self.transforms = transforms
        self.img_list = list()
        for img_path in img_path_list :
            image = Image.open(img_path)
            if self.transforms is not None:
                image = self.transforms(image)

            self.img_list.append(image)
        
    def __getitem__(self, index):
        image = self.img_list[index]
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_list)