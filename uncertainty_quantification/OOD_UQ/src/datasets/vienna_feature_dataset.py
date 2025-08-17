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
import glob
import h5py


class ViennaFeatureDataset(Dataset):
    def __init__(
        self,
        feat_folder="/n/data2/hms/dbmi/kyu/lab/jz290/BRAIN_PM_Vienna_20X_Feats/cTrans_features",
        gbm_inpaths = ['/n/data2/hms/dbmi/kyu/lab/jz290/BRAIN_20X/Ebrain_GBM_all/TILES','/n/data2/hms/dbmi/kyu/lab/jz290/BRAIN_20X/Ebrain_GBM/TILES'],
        pcnsl_inpaths = ['/n/data2/hms/dbmi/kyu/lab/jz290/BRAIN_20X/Ebrain_PCNSL/TILES'],
        filter_file="/n/data2/hms/dbmi/kyu/lab/shl968/pathology_uncertainty-main/Vienna_PICTURE_ID_OOD_predictions/VIenna_ID_tile_preds_top250conf_tiles.csv",
        exclude_uncertain=False, # Filter out uncertain tiles
        transform=None,
        target_transform=None,
        dino=False,
        normalize_stain=False,
        label_map=None,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.exclude_uncertain = exclude_uncertain
        self.dino = dino
        self.gbm_inpaths =gbm_inpaths
        self.pcnsl_inpaths = pcnsl_inpaths
        self.label_map = label_map
        self.dict_label = {
            0: "Leading Edge (LE)",
            1: "Infiltrating Tumor (IT)",
            2: "Hyperplastic blood vessels in infiltrating/cellular tumor (IThbv/CThbv)",
            3: "Cellular Tumor (CT)",
            5: "Perinecrotic Zone (CTpnz)",
            4: "Pseudopalisading cells but no visible necrosis (CTpnn)",
            5: "Pseudopalisading cells around necrosis (CTpan)",
            6: "Microvascular proliferation (CTmvp)",
            7: "Necrosis (CTne)",
            8: "Necrosis (CTne)",
            9: "Background",
        }

        self.reduced_labels = {0: "gbm", 1: "pcnsl"}
        self.normalize_stain = normalize_stain
        self.reduced_labels_reversed = {v: k for k, v in self.reduced_labels.items()}
        self.dict_path_labels = self.find_features(feat_folder, filter_file=filter_file)

        if self.normalize_stain:
            target = transforms.ToTensor()(
                cv2.cvtColor(
                    cv2.imread("src/utils/target_domain.png"), cv2.COLOR_BGR2RGB
                )
            )
            self.stain_norm_ = stain_normalizer(target)

        if self.label_map:
            self.label_map = {
                self.reduced_labels_reversed[label_from]: self.reduced_labels_reversed[
                    label_to
                ]
                for label_from, label_to in label_map.items()
            }

            for i in range(len(self.reduced_labels)):
                if i not in self.label_map.keys():
                    self.label_map[i] = i

            self.label_map_temp = self.label_map.copy()
            for k, v in self.label_map.items():
                self.label_map[k] = self.label_map[k] % len(self.label_map_temp) + 1

            for i in range(len(self.reduced_labels)):
                if i not in self.label_map.values():
                    for k, v in self.label_map.items():
                        if v > i:
                            self.label_map[k] = v - 1
        else:
            self.label_map = {i: i for i in range(5)}

    def __len__(self):
        return len(self.dict_path_labels)
    def get_label(self,idx):
        label = self.dict_path_labels[idx]["coarse_labels"]
        # disabled for now becasue we are using encoded features
        # if self.transform: 
        #     patch = self.transform(image=patch)["image"]



        # disabled for now becasue we are using encoded features
        # if self.normalize_stain: 
        #     patch = self.stain_norm_(patch.mT)

        if self.label_map:
            label_to_use = self.label_map[label]
        else:
            label_to_use = label
        return label_to_use
        

    def __getitem__(self, idx):
        patch_path = self.dict_path_labels[idx]["paths"]
        h5_file = self.dict_path_labels[idx]["h5_file"]
        
        # Read the feature  file from h5 file
        # f = h5py.File(h5_file, 'r')
        # features = f['features'][self.dict_path_labels[idx]["index"]]
        features = self.dict_path_labels[idx]["features"]

        features = torch.tensor(features).view(-1)

        # patch = np.array(Image.open(patch_path))

        label = self.dict_path_labels[idx]["coarse_labels"]
        # disabled for now becasue we are using encoded features
        # if self.transform: 
        #     patch = self.transform(image=patch)["image"]



        # disabled for now becasue we are using encoded features
        # if self.normalize_stain: 
        #     patch = self.stain_norm_(patch.mT)

        if self.label_map:
            label_to_use = self.label_map[label]
        else:
            label_to_use = label

        return {
            "path": patch_path,
            "image": features, # is in fact features in this case. Named image for compatibility with other datasets
            "label": label_to_use,
            "idx": idx,
            "label_name": self.reduced_labels[label],
            "patient_id": self.dict_path_labels[idx]["patient_id"],
            "extra_ood": False,
        }

    def find_features(self, path, labels: list = ["gbm", "pcnsl"],filter_file=None):
        # list all tiles in all h5 files
        dict_path_labels = []
        h5_files = glob.glob(os.path.join(path, "*.h5"))
        gbm_list = [glob.glob(os.path.join(inpath, "*.ndpi")) for inpath in self.gbm_inpaths]
        gbm_list = [item for sublist in gbm_list for item in sublist]
        pcnsl_list = [glob.glob(os.path.join(inpath, "*.ndpi")) for inpath in self.pcnsl_inpaths]
        pcnsl_list = [item for sublist in pcnsl_list for item in sublist]
        # gbm_list = glob.glob(os.path.join(self.gbm_inpath, "*.ndpi"))
        # pcnsl_list = glob.glob(os.path.join(self.pcnsl_inpath, "*.ndpi"))
        gbm_list = [os.path.basename(x).split(".")[0] for x in gbm_list]
        pcnsl_list = [os.path.basename(x).split(".")[0] for x in pcnsl_list]


        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as f:
                array_paths = f["paths"][()]
                array_labels = f["labels"][()]
                array_patient_ids = f["patients"][()]
                features = f['features'][()]
            array_patient_id = array_patient_ids[0].decode("utf-8")
            if array_patient_id in gbm_list:
                label = 'gbm'
            elif array_patient_id in pcnsl_list:
                label = 'pcnsl'
            else:
                # raise ValueError(f"Unknown label for : {array_patient_id}")
                continue
            # label = 'gbm' if array_patient_id in gbm_list else 'pcnsl'
            label = self.reduced_labels_reversed[label]

            for i in range(array_paths.shape[0]):
                dict_path_labels.append(
                    {
                        "h5_file": h5_file,
                        "features": features[i],
                        "index": i,
                        "paths": array_paths[i],
                        "coarse_labels": label,
                        "patient_id": array_patient_ids[i],
                    }
                )

        return dict_path_labels
