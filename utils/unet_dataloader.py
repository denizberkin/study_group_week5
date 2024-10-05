import os
from typing import Tuple, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ISICDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, 
                 transform=None, mask_transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # to exclude superpixels and other unrelated stuff
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]  
        self.transform = transform
        self.mask_transform = mask_transform

    # each torch Dataset must have a __len__ and __getitem__ dunder method implemented.
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        mask_name = img_name.replace('.jpg', '_segmentation.png')  # to find the same mask
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # resize to 256, 256
        image = cv2.resize(image, dsize=(256, 256))
        mask = cv2.resize(mask, dsize=(256, 256)) / 255.

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return {"image": image,
                "mask": mask}  # you can return any metadata that may be used in the network here
        

def get_dataloaders(batch_size: int, shuffle: bool = True, 
                    transform=None, mask_transform=None, num_workers: int = 4
                    ) -> Tuple[DataLoader, DataLoader]:
    """ Creates dataset and dataloader instances for train and validation set 
    Returns: tuple of train_loader and val_loader
    """
    train_set = ISICDataset(image_dir="data/ISIC2017/train_set",
                            mask_dir="data/ISIC2017/train_masks",
                            transform=transform,
                            mask_transform=mask_transform)
    
    val_set = ISICDataset(image_dir="data/ISIC2017/val_set",
                          mask_dir="data/ISIC2017/val_masks",
                          transform=transform,
                          mask_transform=mask_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers)

    return train_loader, val_loader
