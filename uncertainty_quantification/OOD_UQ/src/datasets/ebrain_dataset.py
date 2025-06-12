import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torchvision.transforms import transforms
import pandas as pd
from PIL import Image
import os
import json
import numpy as np
import csv
import cv2
from src.utils.stain_normalizer import stain_normalizer


class EbrainDataset(Dataset):
    def __init__(
        self,
        slides_file="/n/data2/hms/dbmi/kyu/lab/shl968/tile_datasets/EBrain_OOD_1000dense_max500_Q0.95_SY_level1",
        filter_file="/n/data2/hms/dbmi/kyu/lab/shl968/pathology_uncertainty-main/Vienna_PICTURE_ID_OOD_predictions/Vienna_OOD_tile_preds_top250conf_tiles.csv",
        extra_ood_filter_file="/n/data2/hms/dbmi/kyu/lab/shl968/pathology_uncertainty-main/Vienna_PICTURE_ID_OOD_predictions/Vienna_extraOOD_tile_preds_top250conf_tiles.csv",
        exclude_uncertain=False, # Filter out uncertain tiles
        # filter_file = None,
        # extra_ood_filter_file = None,
        transform=None,
        target_transform=None,
        dino=False,
        normalize_stain=False,
        label_map=None,
        seed=0,
        cancer_ood=True,
        extra_ood=False,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.exclude_uncertain = exclude_uncertain
        self.dino = dino
        self.seed = seed
        assert cancer_ood == True or extra_ood == True, \
            "At least one of {cancer_ood, extra_ood } should be True. Otherwise, the dataset will be empty."
        cancer_ood_dict_path_labels  = self.find_jpeg(slides_file,filter_file=filter_file) if cancer_ood else []
        benign_ood_dict_path_labels = self.find_jpeg_extra_ood(filter_file=extra_ood_filter_file) if extra_ood else []
        self.dict_path_labels = cancer_ood_dict_path_labels + benign_ood_dict_path_labels
        print("Number of images in the dataset: ")
        print(f"\tOOD (cancer):\t{len(cancer_ood_dict_path_labels)}")
        print(f"\tOOD (benign):\t{len(benign_ood_dict_path_labels)}")
        print(f"\tTotal:\t{len(self.dict_path_labels)}")


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

        self.reduced_labels = {0: "None"}
        self.normalize_stain = normalize_stain
        self.reduced_labels_reversed = {v: k for k, v in self.reduced_labels.items()}

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

    def __getitem__(self, idx):
        patch_path = self.dict_path_labels[idx]["paths"]
        # Read the png file
        patch = np.array(Image.open(patch_path))

        label = self.dict_path_labels[idx]["coarse_labels"]
        if self.transform:
            patch = self.transform(image=patch)["image"]

        patch = transforms.ToTensor()(patch)

        if self.normalize_stain:
            patch = self.stain_norm_(patch.mT)

        if self.label_map:
            label_to_use = self.label_map[label]
        else:
            label_to_use = label

        return {
            "path": patch_path,
            "image": patch,
            "label": label_to_use,
            "idx": idx,
            "label_name": self.reduced_labels[label],
            "patient_id": self.dict_path_labels[idx]["patient_id"],
            "extra_ood": self.dict_path_labels[idx]["extra_ood"],
        }

    def find_jpeg(self, path, filter_file=None, labels: list = ["Ebrain_OOD"]):
        # Find all the jpeg files in GBM and PCNL folders
        dict_path_labels = []

        # read the csv file that contains a column with the

        # Read the patient IDs from ebrain.csv
        with open("ebrain.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            patient_ids = [row[2].split(".ndpi")[0] for row in reader]

        dict_path_labels = []

        if filter_file is not None and self.exclude_uncertain==True:
            filter_files = pd.read_csv(filter_file)['file']
            filter_slide = [os.path.basename(os.path.dirname(file)) for file in filter_files]
            filter_files = [os.path.basename(file) for file in filter_files]
            df_filter = pd.DataFrame({'file':filter_files,'slide':filter_slide})

        # set random seed
        np.random.seed(self.seed)

        for label_int, label_name in enumerate(labels):
            for root, dirs, files in os.walk(path):
                if files:
                    # Select 25 random files from the list
                    np.random.shuffle(files)
                if os.path.basename(root) == "thumbnails": # skip thumbnails folder
                    continue
                if filter_file is not None and self.exclude_uncertain==True:
                    df_slide = df_filter.loc[df_filter['slide']==os.path.basename(root)]
                    # Keep only the files that are in the filter_file
                    df_slide_certain = df_slide.loc[df_slide['file'].isin(files)]
                    files = df_slide_certain['file'].tolist()



                for file in files[:100]:
                    if file.endswith(".jpg"):
                        # Extract the patient ID from the file path
                        patient_id = os.path.splitext(os.path.basename(root))[0]

                        # Check if the patient ID is in the list
                        if patient_id in patient_ids:
                            dict_path_labels.append(
                                {
                                    "paths": os.path.join(root, file),
                                    "coarse_labels": label_int,
                                    "patient_id": patient_id,
                                    "extra_ood": False,
                                }
                            )

        # Count the number of unique patient_id in dict_path_labels

        return dict_path_labels

    # def exclude_normal(self, df,):
    #     df_anno = pd.read_excel('Vienna_controls_annotation.xlsx',engine='openpyxl')
    #     df_anno_norm = df_anno.loc[df_anno['Description'].str.contains('normal morphology')]
    #     uuid_norm = df_anno_norm['UUID']
    #     df = df.loc[df['uuid'].isin(uuid_norm)]
    #     return df

    def find_jpeg_extra_ood(
        self,
        path="/n/data2/hms/dbmi/kyu/lab/jz290/Ebrains-Control/tile_datasets/testing_SY_1000dense_max500_Q0.95_Zoom20X",
        anno_csv="/n/data2/hms/dbmi/kyu/lab/shl968/pathology_uncertainty-main/VIenna_controls_annotation.csv",
        filter_file=None,
    ):
        # Find all the jpeg files in GBM and PCNL folders
        dict_path_labels = []
        df_anno = pd.read_csv(anno_csv)
        df_anno_norm = df_anno.loc[df_anno['Description'].str.contains('normal morphology')]
        uuid_norm = df_anno_norm['UUID'].tolist()
        
        if filter_file is not None and self.exclude_uncertain==True:
            filter_files = pd.read_csv(filter_file)['file']
            filter_slide = [os.path.basename(os.path.dirname(file)) for file in filter_files]
            filter_files = [os.path.basename(file) for file in filter_files]
            df_filter = pd.DataFrame({'file':filter_files,'slide':filter_slide})

        for root, dirs, files in os.walk(os.path.join(path)):
            if os.path.basename(root) == "thumbnails": # skip thumbnails folder
                continue
            ## skip if not in the normal list
            if os.path.basename(root).replace('.ndpi','') not in uuid_norm:
                continue

            if filter_file is not None and self.exclude_uncertain==True:
                df_slide = df_filter.loc[df_filter['slide']==os.path.basename(root)]
                # Keep only the files that are in the filter_file
                df_slide_certain = df_slide.loc[df_slide['file'].isin(files)]
                files = df_slide_certain['file'].tolist()

            for file in files:
                if file.endswith(".jpg"):
                    dict_path_labels.append(
                        {
                            "paths": os.path.join(root, file),
                            "coarse_labels": 0,
                            "patient_id": os.path.splitext(os.path.basename(root))[0],
                            "extra_ood": True,
                        }
                    )

        return dict_path_labels
