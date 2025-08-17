from cgi import test
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from src.datasets.slides_dataset import SlidesDataset

IMAGE_SIZE = 100

class DinoDataset(SlidesDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.randcrop_big = transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.5, 1.0))
        self.randcrop_small = transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE))
        
    def __getitem__(self, idx):
        patch, _ = super().__getitem__(idx)
        patch = transforms.ToTensor()(patch)
        small, big = self.randcrop_small(patch), self.randcrop_big(patch)
        return small, big