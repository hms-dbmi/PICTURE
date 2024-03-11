import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torchvision.transforms import transforms
from src.utils.stain_normalizer import stain_normalizer
from collections import defaultdict
import os
import json
import numpy as np
import h5py
import pickle
import cv2
class BrainDataset(Dataset):
    def __init__(self, slides_file="/home/cl427/GBM/results/train_vs_test_by_pt/subsample_0326.hdf5", stage = "train", transform=None, target_transform=None, dino=False, p_uncertainty=0.0, normalize_stain = False, label_to_use = None, label_map = None):
        
        self.transform = transform
        self.target_transform = target_transform
        self.dino = dino
        self.array_images, self.array_labels = self.read_h5(slides_file, stage)
        self.p_uncertainty = p_uncertainty
        self.normalize_stain = normalize_stain
        self.label_to_use = label_to_use
        self.label_map = label_map
        
        self.dict_label = {0: "Leading Edge (LE)", 1: "Infiltrating Tumor (IT)",
            2: "Hyperplastic blood vessels in infiltrating/cellular tumor (IThbv/CThbv)", 3: "Cellular Tumor (CT)", 5: "Perinecrotic Zone (CTpnz)",
            4: "Pseudopalisading cells but no visible necrosis (CTpnn)", 5: "Pseudopalisading cells around necrosis (CTpan)",
            6: "Microvascular proliferation (CTmvp)", 7: "Necrosis (CTne)",
            8: "Necrosis (CTne)", 9: "Background"}

        self.reducing = {0:0, 1:1, 2:2, 3:2, 4:3, 5:3, 6:3, 7:3, 8:3, 9:4}
        self.reduced_labels = {0: "Leading Edge (LE)", 1: "Infiltrating Tumor (IT)", 2: "Cellular Tumor (CT)", 3: "Necrosis (CTne)", 4: "Background"}
        self.reduced_labels_reversed = {v: k for k, v in self.reduced_labels.items()}
        
        if self.p_uncertainty > 0:

            # Read uncertainty dictionary, with key the index of the patch and value the uncertainty
            if os.path.exists("data/processed/uncertainty.pickle"):
                with open("data/processed/uncertainty.pickle", "rb") as f:
                    self.uncertainty = pickle.load(f)
            else:
                # throw error
                pass

            self.uncertainty_filter(self.uncertainty, self.p_uncertainty)

        if self.normalize_stain:
            target = transforms.ToTensor()(cv2.cvtColor(cv2.imread("src/utils/target_domain.png"), cv2.COLOR_BGR2RGB))
            self.stain_normalizer_ = stain_normalizer(target)

        if label_to_use:
            # If binary_label is not in self.reduce_labels values then it will throw an error
            if np.in1d(label_to_use, list(self.reduced_labels.values())).sum() != len(label_to_use) :
                raise ValueError(f"Some of the labels in {label_to_use} are not in {self.reduced_labels.values()}")

            self.label_to_use = {self.reduced_labels_reversed[label]:i for i,label in enumerate(self.label_to_use)}
            last_label = len(self.label_to_use)
            self.label_to_use = defaultdict(lambda: last_label, self.label_to_use)
        else:
            self.label_to_use = {i:i for i in range(5)}

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



    def uncertainty_filter(self, uncertainty_per_patch = None, p_uncertainty = None ):
        """ Filter the dataset based on the uncertainty of the patches. The uncertainty is computed using the epistemic confidence. 
        Args:
            uncertainty_per_patch (dict): dictionary with key the index of the patch and value the uncertainty
            p_uncertainty (float): quantile of the uncertainty distribution to filter the patches
        
        Returns:
            None
        
        """ 

        # Uncertainty distribution per class
        self.uncertainty_per_class = {i: {k:v["aleatoric_confidence"] for k,v in self.uncertainty.items() if v["targets"] == i} for i in range(5)}

        # For every class, compute the quantile of the uncertainty distribution
        self.quantiles_per_class = {i: np.quantile(list(self.uncertainty_per_class[i].values()), self.p_uncertainty) for i in range(5)}

        # Compute the indices of the patches with uncertainty above the quantile
        self.indices = [k for k,v in self.uncertainty.items() if v["aleatoric_confidence"] > self.quantiles_per_class[v["targets"]]]

        # Filter the images and labels
        self.array_images = self.array_images[self.indices]
        self.array_labels = self.array_labels[self.indices]


    def __len__(self):
        return len(self.array_labels)

    def __getitem__(self, idx):
        patch = self.array_images[idx]
        label = self.array_labels[idx]
        # Reduce the number of classes to 5
        label = self.reducing[label]

        if self.transform:
            patch = self.transform(image = patch)['image']
        if self.dino:
            small = transforms.RandomResizedCrop(
                (self.crop_size, self.crop_size))(patch)
            big = transforms.RandomResizedCrop(
                (self.crop_size, self.crop_size), scale=(0.5, 1.0))(patch)
            patch = (small, big)

        patch = transforms.ToTensor()(patch)

        if self.normalize_stain:
            patch = self.stain_normalizer_(patch.mT)

        label_to_use = self.label_to_use[label]

        if self.label_map:
            label_to_use = self.label_map[label_to_use]

        return {"image": patch, "label": label_to_use, "idx": idx, "label_name": self.reduced_labels[label]}

    def read_h5(self, path, stage):
        with h5py.File(path, "r") as f:
            array_images = f[f'{stage}_img'][()] 
            array_labels = f[f'{stage}_labels'][()]

        return array_images, array_labels
