import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
# from torchvision import transforms
import  torchvision.transforms.v2 as v2
from typing import List,Literal
import torch
import torch.nn as nn

import torchstain
import cv2
from utils.macenko_mod import TorchMacenkoNormalizer
from timm.data.transforms_factory import create_transform
# def pre_transforms():

class TorchBatchMacenkoNormalizer(nn.Module):
    '''
    nn module for performing Macenko normalization on a batch of images
    '''
    def __init__(self,Io=240, alpha=1, beta=0.15,
                 source_thumbnail=None,
                 source_thumbnail_mask=None):
        '''
        Io, alpha, beta: parameters for Macenko normalization
        source_thumbnail: thumbnail of the source image to be used for normalization
        source_thumbnail_mask: mask of the source image to be used for normalization
        '''
        super().__init__()  
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        self.normalizer = TorchMacenkoNormalizer()
        
        
        if source_thumbnail is not None:
            source_thumbnail = Image.open(source_thumbnail).convert('RGB')
            source_thumbnail = np.array(source_thumbnail)
            if source_thumbnail_mask is not None:
                source_thumbnail_mask = Image.open(source_thumbnail_mask).convert('RGB')
                source_thumbnail_mask = np.array(source_thumbnail_mask)
                source_thumbnail = source_thumbnail[np.max(source_thumbnail_mask,axis=2)>0,:]
                # add dimension at the beginning
                source_thumbnail = source_thumbnail[np.newaxis,...]
            source_thumbnail = torch.tensor(source_thumbnail).permute(2,0,1)
            self.HE, _ = self.normalizer.fit_source(source_thumbnail,Io=Io,alpha=alpha,beta=beta)
        else:
            self.HE = None

        
        
    def to(self,device):
        # self.normalizer.to(device)
        
        self.normalizer.HERef = self.normalizer.HERef.to(device)
        self.normalizer.maxCRef = self.normalizer.maxCRef.to(device)
        if self.HE is not None:
            self.HE = self.HE.to(device)

        return super().to(device)
    def forward(self,x):
        '''
        x: torch.Tensor of shape (B,C,H,W), or (C,H,W)
        For the normalizer to work on 4D tensor, we need to flatten the tensor to 3D tensor
        '''
        if x.dim() == 4:
            B,C,H,W = x.shape
            x = x.permute(1,2,3,0).reshape(C,H,W*B) # convert to 3D tensor (C,H,W*B)
            x_norm,_,_ = self.normalizer.normalize(
                x,Io=self.Io, alpha=self.alpha,beta=self.beta,HE=self.HE,stains=False) # output shape (H,W*B,C)
            x_norm = x_norm.reshape(H,W,B,C).permute(2,3,0,1) # convert back to 4D tensor (B,C,H,W)
        elif x.dim() == 3:
            x_norm,_,_ = self.normalizer.normalize(
                x,Io=self.Io, alpha=self.alpha,beta=self.beta,HE=self.HE,stains=False) # output shape (H,W,C)
            x_norm = x_norm.permute(2,0,1) # convert back to 3D tensor (C,H,W)
            
            
        return x_norm
        



def get_transforms_albumentation(train=False):
    """
    Takes a list of images and applies the same augmentations to all of them.
    This is completely overengineered but it makes it easier to use in our pipeline
    as drop-in replacement for torchvision transforms.

    ## Example

    ```python
    imgs = [Image.open(f"image{i}.png") for i in range(1, 4)]
    t = get_transforms(train=True)
    t_imgs = t(imgs) # List[torch.Tensor]
    ```

    For the single image case:

    ```python
    img = Image.open(f"image{0}.png")
    # or img = np.load(some_bytes)
    t = get_transforms(train=True)
    t_img = t(img) # torch.Tensor
    ```
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    _data_transform = None

    def _get_transform(n: int = 3):
        if train:
            data_transforms = A.Compose(
                [
                    A.Resize(224, 224),
                    A.RandomResizedCrop(224, 224, scale=(0.2, 1.0)),
                    A.HorizontalFlip(),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ],
                additional_targets={f"image{i}": "image" for i in range(1, n)},
            )
        else:
            data_transforms = A.Compose(
                [
                    A.Resize(224, 224),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ],
                additional_targets={f"image{i}": "image" for i in range(1, n)},
            )
        return data_transforms

    def transform_images(images: any):
        nonlocal _data_transform

        if not isinstance(images, list):
            n = 1
            images = [images]
        else:
            n = len(images)
        if _data_transform is None:
            # instantiate once
            _data_transform = _get_transform(n)

        # accepts both lists of np.Array and PIL.Image
        if isinstance(images[0], Image.Image):
            images = [np.array(img) for img in images]

        image_dict = {"image": images[0]}
        for i in range(1, n):
            image_dict[f"image{i}"] = images[i]

        transformed = _data_transform(**image_dict)
        transformed_images = [
            transformed[key] for key in transformed.keys() if "image" in key
        ]

        if len(transformed_images) == 1:
            return transformed_images[0]
        return transformed_images

    return transform_images

def get_transforms_timm(trans_dict={}):
    if len(trans_dict) == 0:
        trans_dict = {
            'input_size': (3, 224, 224),
             'interpolation': 'bicubic',
             'mean': (0.485, 0.456, 0.406),
             'std': (0.229, 0.224, 0.225),
             'crop_pct': 1.0,
             'crop_mode': 'center'}
    return create_transform(**trans_dict)