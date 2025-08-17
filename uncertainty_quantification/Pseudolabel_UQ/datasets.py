
import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ExponentialLR
import torchmetrics as tm
import os


from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from PIL import Image
import numpy as np


class TileDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None,use_gpu=False):
        self.img_labels = list(df['label'])
        self.img_dir = list(df['file'])
        self.transform = transform
        self.target_transform = target_transform
        self.pre_transform = transforms.ToTensor()
        self.use_gpu = use_gpu

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = Image.open(img_path)
        label = self.img_labels[idx]
        image = self.pre_transform(image)
        if self.use_gpu:
            image = image.to('cuda')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_path


