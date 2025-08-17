import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torchvision.transforms import transforms
import pandas as pd
from PIL import Image
import os
import json
import numpy as np
import h5py
import cv2
from src.utils.stain_normalizer import stain_normalizer

class UPennDataset(Dataset):
    def __init__(self, slides_file="/n/data2/hms/dbmi/kyu/lab/datasets/UPenn/annotated_coarse_pickles/fg_anno_patition_0.csv", stage = "train", transform=None, target_transform=None, dino=False, normalize_stain = False, label_map = None):
        
        self.transform = transform
        self.target_transform = target_transform
        self.dino = dino
        self.dict_path_labels = self.read_csv(slides_file, stage)
        self.label_map = label_map
        self.dict_label = {0: "Leading Edge (LE)", 1: "Infiltrating Tumor (IT)",
            2: "Hyperplastic blood vessels in infiltrating/cellular tumor (IThbv/CThbv)", 3: "Cellular Tumor (CT)", 5: "Perinecrotic Zone (CTpnz)",
            4: "Pseudopalisading cells but no visible necrosis (CTpnn)", 5: "Pseudopalisading cells around necrosis (CTpan)",
            6: "Microvascular proliferation (CTmvp)", 7: "Necrosis (CTne)",
            8: "Necrosis (CTne)", 9: "Background"}

        self.reduced_labels = {0: "Necrosis", 1: "Glioma Squash", 2: "Macrophages", 3: "Meninges", 4: "Artifact"}
        self.normalize_stain = normalize_stain
        self.reduced_labels_reversed = {v: k for k, v in self.reduced_labels.items()}


        if self.normalize_stain:
            target = transforms.ToTensor()(cv2.cvtColor(cv2.imread("src/utils/target_domain.png"), cv2.COLOR_BGR2RGB))
            self.stain_norm_ = stain_normalizer(target)


        if self.label_map:
            self.label_map = {self.reduced_labels_reversed[label_from]:  self.reduced_labels_reversed[label_to]  for label_from, label_to in label_map.items()}
            
            for i in range(len(self.reduced_labels)):
                if i not in self.label_map.keys():
                    self.label_map[i] = i
    
            self.label_map_temp = self.label_map.copy()
            for k,v in self.label_map.items():
                self.label_map[k] = self.label_map[k] % len(self.label_map_temp) +1

            for i in range(len(self.reduced_labels)):
                if i not in self.label_map.values():
                    for k,v in self.label_map.items():
                        if v > i:
                            self.label_map[k] = v-1
        else:
            self.label_map = {i:i for i in range(5)}


    def __len__(self):
        return len(self.dict_path_labels)

    def __getitem__(self, idx):
        patch_path = self.dict_path_labels[idx]["paths"]
        # Read the png file
        patch = np.array(Image.open(patch_path))

        label = self.dict_path_labels[idx]["coarse_labels"]
        if self.transform:
            patch = self.transform(image = patch)['image']
            

        patch = transforms.ToTensor()(patch)

        if self.normalize_stain:
            patch = self.stain_norm_(patch.mT)

        if self.label_map:
            label_to_use = self.label_map[label]
        else:
            label_to_use = label

        return {"image": patch, "label": label_to_use, "idx": idx, "label_name": self.reduced_labels[label]}

    def read_h5(self, path, stage):
        with h5py.File(path, "r") as f:
            array_images = f[f'{stage}_images'][()] 
            array_labels = f[f'{stage}_coarse_labels'][()]

        return array_images, array_labels

    def read_csv(self, path, stage):
        if stage == "val":
            stage = "test"

        df = pd.read_csv(path)
        # Make a dict with key the index and value the paths
        return df[df["partition"] == stage].to_dict(orient="index")
