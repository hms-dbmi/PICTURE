import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torchvision.transforms import transforms
import os
import json
import numpy as np
import h5py


class CellphoneDataset(Dataset):
    def __init__(self, slides_file="/n/data2/hms/dbmi/kyu/lab/jz290/cell-phone/RCC_BWH_photo.hdf5", transform=None, target_transform=None, dino=False):
        
        self.transform = transform
        self.target_transform = target_transform
        self.dino = dino
        self.array_images, self.array_labels = self.read_h5(slides_file)

    def __len__(self):
        return len(self.array_labels)

    def __getitem__(self, idx):
        patch = self.array_images[idx]
        label = self.array_labels[idx]
        if self.transform:
            patch = self.transform(image = patch)['image']
        if self.dino:
            small = transforms.RandomResizedCrop(
                (self.crop_size, self.crop_size))(patch)
            big = transforms.RandomResizedCrop(
                (self.crop_size, self.crop_size), scale=(0.5, 1.0))(patch)
            patch = (small, big)

        patch = transforms.ToTensor()(patch)
        return {"image": patch, "label": label, "idx": idx, "label_name": str(label)}

    def read_h5(self, path):
        with h5py.File(path, "r") as f:
            array_images = f["img"][()]
            array_labels = f["subtype"][()]

        return array_images, array_labels
