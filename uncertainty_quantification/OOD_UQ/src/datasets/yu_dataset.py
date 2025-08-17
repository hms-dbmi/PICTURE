import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torchvision.transforms import transforms
import os
import json
import numpy as np
import h5py
import glob
from PIL import Image
import albumentations as A

class YuDataset(Dataset):
    def __init__(self, slides_file="data/Inflammation2_665716", stage = "train", transform=None, target_transform=None):
        
        self.transform = transform
        self.target_transform = target_transform
        self.array_images, self.array_labels = self.find_files(slides_file)

    def __len__(self):
        return len(self.array_labels)

    def __getitem__(self, idx):
        patch = np.array(Image.open(self.array_images[idx]))
        label = np.array(Image.open(self.array_labels[idx]))
        
        if self.transform:
            patch = self.transform(image = patch)['image']
        if self.target_transform:
            transformed = self.target_transform(image = patch, mask = label)
            patch, label = transformed['image'], transformed['mask']

        patch = transforms.ToTensor()(patch)
        label = torch.Tensor(label).int()
        
        return patch, label

    def find_files(self, path):
        
        labels = glob.glob(os.path.join(path,'*/distance*.jpg'))
        images = labels.copy()
        return images, labels
