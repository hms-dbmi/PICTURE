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

class WholeSlideDataset(Dataset):
    def __init__(self, slides_file="/n/data2/hms/dbmi/kyu/lab/datasets/IvyGap/compressed", stage = "train", transform=None, target_transform=None, dino=False):
        
        self.transform = transform
        self.target_transform = target_transform
        self.dino = dino
        self.array_images, self.array_labels = self.find_files(slides_file)
        self.dict_label = {0: "Leading Edge (LE)", 1: "Infiltrating Tumor (IT)",
            2: "Hyperplastic blood vessels in infiltrating/cellular tumor (IThbv/CThbv)", 3: "Cellular Tumor (CT)", 5: "Perinecrotic Zone (CTpnz)",
            4: "Pseudopalisading cells but no visible necrosis (CTpnn)", 5: "Pseudopalisading cells around necrosis (CTpan)",
            6: "Microvascular proliferation (CTmvp)", 7: "Necrosis (CTne)",
            8: "Necrosis (CTne)", 9: "Background"}
        self.mapping = {(33, 143, 166): 0, (210,5,208): 1, (5, 208, 4): 3, (37, 209, 247): 5, (6, 208,170): 4, (255, 102, 0): 6, (5,5,5): 8, (255, 255,255): 9}
        self.reducing = {0:0, 1:1, 2:2, 3:2, 5:3, 4:3, 6:3, 7:7, 8:3, 9:4}
        self.reduced_labels = {0: "Leading Edge (LE)", 1: "Infiltrating Tumor (IT)", 2: "Cellular Tumor (CT)", 3: "Necrosis (CTne)", 4: "Background"}
        
    def __len__(self):
        return len(self.array_labels)

    def __getitem__(self, idx):
        patch = np.array(Image.open(self.array_images[idx]))
        label = np.array(Image.open(self.array_labels[idx]))[:,:,0]
        
        if self.transform:
            patch = self.transform(image = patch)['image']
        if self.target_transform:
            transformed = self.target_transform(image = patch, mask = label)
            patch, label = transformed['image'], transformed['mask']
        if self.dino:
            small = transforms.RandomResizedCrop(
                (self.crop_size, self.crop_size))(patch)
            big = transforms.RandomResizedCrop(
                (self.crop_size, self.crop_size), scale=(0.5, 1.0))(patch)
            patch = (small, big)


        patch = transforms.ToTensor()(patch)
        label = torch.Tensor(label).int()
        return patch, label

    def find_files(self, path):
        
        labels = glob.glob(os.path.join(path,'*/*A.png'))
        images = [label.replace('A.png', '.png') for label in labels]

        return images, labels
