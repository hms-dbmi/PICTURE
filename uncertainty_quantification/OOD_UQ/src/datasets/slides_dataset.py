import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torchvision.transforms import transforms
import os
import json
import pyvips
import numpy as np


class SlidesDataset(Dataset):
    def __init__(self, slides_file = 'data/slides.json', patch_per_slide=1, crop_size = 300, patch_size = 224, transform=None, target_transform=None, dino=False):
        self.slides_file = self.read_list_slides(slides_file)
        self.patch_per_slide = patch_per_slide
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform
        self.dino = dino

    def __len__(self):
        return len(self.slides_file)*self.patch_per_slide

    def __getitem__(self, idx):
        img_path = self.slides_file[idx // self.patch_per_slide]['path']
        patch = self.read_slide(img_path)
        label = int(self.slides_file[idx // self.patch_per_slide]['outcome'] == 'Alive')
        if self.transform:
            patch = self.transform(image = patch)['image']
        if self.target_transform:
            label = self.target_transform(label)
        if self.dino:
            small =  transforms.RandomResizedCrop((self.crop_size, self.crop_size))(patch)
            big = transforms.RandomResizedCrop((self.crop_size, self.crop_size), scale=(0.5, 1.0))(patch)
            patch = (small, big)

        return patch, label

    def read_list_slides(self,slides_file):
        with open(f'{slides_file}') as handle:
            list = json.loads(handle.read())
        return list

    def read_slide(self, slide_path):
        # Read entire slide
        slide = pyvips.Image.new_from_file(slide_path)
        height = slide.height
        width = slide.width
        length = self.crop_size

        # Crop a random patch of the slide that is not empty
        std = 0
        while std < 20:
            random_left = np.random.randint(length, height-length)
            random_top = np.random.randint(length, width-length)
            region = pyvips.Region.new(slide)
            bytes = region.fetch(random_top, random_left, length, length)
            patch = np.ndarray(buffer=bytes, dtype=np.uint8,
                               shape=[length, length, 4])[:, :, :3]
            std = patch.std()

        patch = interpolate(torch.tensor(patch.T).unsqueeze(
            0), size=(self.patch_size, self.patch_size)).squeeze().numpy().T
        return patch
