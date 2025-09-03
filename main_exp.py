import pandas as pd
import numpy as np
import os
import random
from glob import glob
import h5py
from ctran import ctranspath
import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import os.path
import tables
import json
import timm
import openslide

from timm.models.layers.helpers import to_2tuple
import timm
import torch.nn as nn
import torch
import torchvision
import os


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def ctranspath():
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    return model


def main(MODEL_PATH = None):

    device_id = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = ctranspath()
    model.head = nn.Identity()
    linear_model = LinearClassifier(input_dim=768)

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        td = torch.load(r'ctranspath.pth')
        model.load_state_dict(td['model'], strict=True)
        linear_model.load_state_dict(torch.load(MODEL_PATH))

    else:
        td = torch.load(r'ctranspath.pth',map_location=torch.device('cpu'))
        model.load_state_dict(td['model'], strict=True)
        linear_model.load_state_dict(torch.load(MODEL_PATH),map_location=torch.device('cpu'))

    model.head = linear_model

    model.to(device)
    print(model)

    print('model is loaded!')

if __name__ == '__main__':

    main(MODEL_PATH = 'binary_pm.pth')

