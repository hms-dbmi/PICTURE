import os
from collections import Counter, defaultdict
from random import sample
from typing import Callable, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import sqlite3
from utils.constants import EMBEDDING_SIZES
from PIL import Image
from utils.helpers import get_transforms


def load_success_ids(feat_folder: str):
    """
    Backwards-compatible loading of success IDs.
    We either load the available slide ids from the deprecated success.txt file
    or from the success.db sqlite database.

    If both files exist, we always prefer the database.
    """
    success_ids = set()
    success_txt = f"{feat_folder}/success.txt"
    success_db = f"{feat_folder}/success.db"
    if os.path.exists(success_txt):
        print("Warning: Loading success IDs from deprecated success.txt.")
        with open(success_txt, "r") as f:
            for line in f:
                success_ids.add(line.strip())
    if os.path.exists(success_db):
        print("Loading success IDs from database.")
        conn = sqlite3.connect(success_db)
        cursor = conn.cursor()
        cursor.execute("SELECT slide_id FROM success")
        success_ids = set([row[0] for row in cursor.fetchall()])
        conn.close()
    return success_ids


class MultiExpertDataset(Dataset):
    """
    This dataset loads and processes features from multiple experts, handling scenarios where different experts
    may provide distinct views or representations of the data. Each slide can have multiple patches and the dataset
    allows for variable numbers of patches per slide, either through random selection or by choosing the top N patches
    based on their tissue density, as measured by the tissue percentage metadata column in the corresponding `slide_id.csv`.

    To understand the dataset loading logic, please refer to the `create_tiles.py` and `create_features.py` scripts.
    Those have been used to extract the raw tiles and features for each expert from each tile.

    Each item returned by this dataset includes:
        - A list of tensors, where each tensor represents the embeddings for a slide's patches as processed by a different expert model.
        - A list of tensors, where each tensor is a boolean mask indicating whether the corresponding patch was padded.
        - The label for the whole slide

    Optionally, it can also return:
        - Patch positions and tissue percentages as metadata for each patch.

    Parameters:
        csv_path (str): Path to the csv file containing metadata about slides.
        patch_folder (str): Directory containing patch data for each slide.
        feat_folder (str): Directory containing feature embeddings from the experts for each slide.
        n_patches (int, optional): Number of patches to include per slide; if None, all patches are used => may cause issues with your dataloader.
        get_metadata (bool): Flag to indicate whether additional metadata should be returned.
        random_selection (bool): If True, patches are randomly selected up to n_patches; otherwise, the top n_patches are selected based on tissue percentage.
        limit (int, optional): If set, limits the number of slides loaded to this number.
        magnification (int): Magnification level used for the features, typical values are 20, or 40.
        experts (List[str]): List of experts (model names) whose features are to be used.
        wsi_type (str): Type of slide preparation, e.g., 'permanent' or 'frozen'. May be ignored, if the column is not present in the csv.
        verbose (bool): If True, prints detailed logs and analyses of the dataset during initialization.

    Example:
        dataset = MultiExpertDataset(
            csv_path="data/train.csv",
            patch_folder="data/patches",
            feat_folder="data/features",
            n_patches=100,
            random_selection=True,
            magnification=40,
            experts=["ctrans", "uni", "swin224"]
        )
        features, masks, label = dataset[0]
        for feature in features:
            print(feature.shape)
        # > torch.Size([100, 768])
        # > torch.Size([100, 1024])
        # > torch.Size([100, 2048])
    """

    def __init__(
        self,
        csv_path: str,
        patch_folder: str,
        feat_folder: str,
        n_patches=None,
        get_metadata=False,
        random_selection=False,
        limit=None,
        magnification=40,
        experts: List[str] = ["ctrans", "uni", "swin224"],
        wsi_type: str = "permanent",
        verbose: bool = False,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.patch_folder = patch_folder
        self.feat_folder = feat_folder
        self.n_patches = n_patches
        self.get_metadata = get_metadata
        self.random_selection = random_selection
        self.limit = limit
        self.magnification = magnification
        self.label_column = "label"
        self.experts = [expert.upper() for expert in experts]
        self.wsi_type = str(wsi_type).lower()
        self.verbose = verbose

        self.csv = pd.read_csv(csv_path)
        if "wsi_type" in self.csv.columns:
            self.csv = self.csv[self.csv["wsi_type"] == self.wsi_type]
        else:
            print("Warning: No 'wsi_type' column found in csv!")
        self.csv = self.csv[self.csv["uuid"].isin(os.listdir(self.feat_folder))]

        self.total_slides = len(self.csv["uuid"].unique())
        self.slide_ids = self.csv["uuid"].unique()
        self.slide_ids = [
            x for x in self.slide_ids if os.path.exists(f"{self.patch_folder}/{x}.csv")
        ]

        success_ids = load_success_ids(self.feat_folder)
        self.slide_ids = [x for x in self.slide_ids if x in success_ids]

        if self.limit is not None:
            self.slide_ids = self.slide_ids[: self.limit]

        self.init_labels()
        self.calculate_class_counts()
        self.label_counts = Counter(self.csv[self.label_column])

    def calculate_class_counts(self):
        """
        Calculates the number of samples for each class.
        """
        self.class_counts = self.csv[self.label_column].value_counts().to_dict()

    def __len__(self):
        return len(self.slide_ids)

    def init_labels(self):
        unique_labels = self.csv[self.label_column].unique()
        self.labels = list(unique_labels)
        self.total_labels = len(self.labels)

    def get_label(self, slide_id):
        return self.csv.loc[self.csv["uuid"] == slide_id, self.label_column].values[0]

    def __getitem__(self, patch_idx):
        slide_id = self.slide_ids[patch_idx]
        patches_csv = pd.read_csv(f"{self.patch_folder}/{slide_id}.csv")
        if self.random_selection and self.n_patches is not None:
            patches_csv = patches_csv.sample(n=min(len(patches_csv), self.n_patches))
        elif not self.random_selection and self.n_patches is not None:
            patches_csv = patches_csv.sort_values(
                by=["tissue_percentage"], ascending=False
            ).iloc[: self.n_patches]

        patch_idx = patches_csv["idx"].values

        embeddings_list = []
        loaded_embeddings = {}
        mask_list = []
        for expert in self.experts:
            if (slide_id, expert) not in loaded_embeddings:
                embeddings, indices = self.load_embeddings(
                    slide_id, expert, self.magnification
                )
                loaded_embeddings[(slide_id, expert)] = (embeddings, indices)
            embeddings, indices = loaded_embeddings[(slide_id, expert)]

            embeddings_for_patches = []
            padded_indices = []

            for i, p_idx in enumerate(patch_idx):
                if p_idx in indices:
                    embeddings_for_patches.append(embeddings[indices[p_idx]])
                else:
                    zero_tensor = torch.zeros_like(embeddings[0])
                    embeddings_for_patches.append(zero_tensor)
                    padded_indices.append(i)

            # pad the embeddings if required
            target_size = EMBEDDING_SIZES[str(expert).lower()]  # eg [768] for ctrans
            # print(f"Target size for {expert}: {target_size}")

            additional_patch_count = self.n_patches - len(embeddings_for_patches)
            if additional_patch_count > 0:
                additional_patches = [
                    torch.zeros(target_size) for _ in range(additional_patch_count)
                ]
                embeddings_for_patches.extend(additional_patches)
                padded_indices.extend(
                    range(
                        len(embeddings_for_patches) - additional_patch_count,
                        len(embeddings_for_patches),
                    )
                )

            embeddings_list.append(torch.stack(embeddings_for_patches))

            # create a boolean mask based on the padded_indices
            mask = torch.zeros(len(embeddings_for_patches), dtype=torch.bool)
            mask[padded_indices] = True
            mask = mask.reshape(-1, 1)
            mask_list.append(mask)

        label = self.get_label(slide_id)

        if self.get_metadata:
            patch_positions = patches_csv[["x", "y"]].values
            tissue_percentages = patches_csv["tissue_percentage"].values
            return (
                embeddings_list,
                mask_list,
                patch_positions,
                tissue_percentages,
                slide_id,
                label,
            )
        return embeddings_list, mask_list, label

    def load_embeddings(self, slide_id, model_type, magnification):
        embed_file = f"{self.feat_folder}/{slide_id}/{magnification}x_{model_type}.npy"
        index_file = f"{self.feat_folder}/{slide_id}/{magnification}x_{model_type}.txt"

        embeddings = np.load(embed_file)
        with open(index_file, "r") as f:
            indices = [int(line.strip()) for line in f.readlines()]
        indices = {idx: i for i, idx in enumerate(indices)}

        return torch.from_numpy(embeddings), indices

    def plot_label_distribution(self):
        self.csv[self.label_column].value_counts().plot(kind="bar")


# ================== Padding Verification ==================
# To verify that the mask lists are correct, you can use the following code:
#
#
# n_patches = 20000 # some large number you know is greater than the number of patches
# ds = MultiExpertDataset(
#     csv_path="../data/tumor_type/lgg/5-fold/train_fold_0.csv",
#     patch_folder="/media/chris/Elements/tcga-lgg/permanent_patches",
#     feat_folder="/media/chris/Elements/tcga-lgg/permanent_embeddings",
#     random_selection=True,
#     n_patches=n_patches,
#     limit=None,
#     magnification=20,
#     label_column="Tumor Type",
#     experts=["ctrans", "chief", "resnet50"],
# )
# batch = ds[0]
# embeddings_list, mask_list, label = batch
# for embeddings, mask in zip(embeddings_list, mask_list):
#     for i, (embed, is_padded) in enumerate(zip(embeddings, mask)):
#         if is_padded:
#             assert torch.all(
#                 embed == 0
#             ), f"Error in expert {i}: Mask marked as padded but embedding is not zero."
#         else:
#             assert not torch.all(
#                 embed == 0
#             ), f"Error in expert {i}: Embedding is zero but mask is not marked as padded."


class H5MultiExpertDatasetInMemory(Dataset):
    """
    This dataset class loads and processes features from multiple experts stored in h5 files,
    keeping all the features in memory for fast access.

    !! This requires a lot of memory, so use with caution !!

    Each item returned by this dataset includes:
        - A list of tensors, where each tensor represents the embeddings for a slide's patches
          as processed by a different expert model.
        - A list of tensors, where each tensor is a boolean mask indicating whether the
          corresponding patch was padded.
        - The label for the whole slide

    Parameters:
        csv_path (str): Path to the csv file containing metadata about slides.
        feat_folder (str): Directory containing feature embeddings from the experts for each slide.
        n_patches (int, optional): Number of patches to include per slide. If None, all patches are used.
        magnification (int): Magnification level used for the features, typical values are 20, or 40.
        label_column (str): Name of the column in the csv that contains the labels.
        experts (List[str]): List of experts (model names) whose features are to be used.
        wsi_type (str): Type of slide preparation, e.g., 'permanent' or 'frozen'. May be ignored
                        if the column is not present in the csv.
        verbose (bool): If True, prints detailed logs and analyses of the dataset during initialization.
    """

    def __init__(
        self,
        csv_path: str,
        feat_folder: str,
        n_patches=None,
        magnification=40,
        experts: List[str] = ["ctrans", "uni", "swin224"],
        wsi_type: str = "permanent",
        verbose: bool = False,
        get_metadata=False,
        random_selection=False,
        with_clinical: bool = False,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.feat_folder = feat_folder
        self.n_patches = n_patches
        self.magnification = magnification
        self.label_column = "label"
        self.experts = [expert.upper() for expert in experts]
        self.wsi_type = str(wsi_type).lower()
        self.verbose = verbose
        self.get_metadata = get_metadata
        self.random_selection = random_selection
        self.with_clinical = with_clinical
        # for now hardcoded
        self.clinical_columns = [
            "Age",
            "Sex",
            "Race_American Indian or Alaska Native",
            "Race_Asian",
            "Race_Black or African American",
            "Race_White",
        ]

        self.csv = pd.read_csv(csv_path)
        if "wsi_type" in self.csv.columns:
            self.csv = self.csv[self.csv["wsi_type"] == self.wsi_type]
        else:
            print("Warning: No 'wsi_type' column found in csv!")
        self.csv = self.csv[self.csv["uuid"].isin(os.listdir(self.feat_folder))]

        if with_clinical:
            # verify that all columns are in self.csv
            for col in self.clinical_columns:
                if col not in self.csv.columns:
                    raise ValueError(
                        f"Clinical data column {col} was not found in csv: {csv_path}"
                    )

        self.total_slides = len(self.csv["uuid"].unique())
        self.slide_ids = self.csv["uuid"].unique()

        success_ids = load_success_ids(self.feat_folder)
        self.slide_ids = [x for x in self.slide_ids if x in success_ids]

        self.init_labels()
        self.calculate_class_counts()
        self.label_counts = Counter(self.csv[self.label_column])
        self.features_in_memory = self._load_features_into_memory()
        self.compute_weights()

    def compute_weights(self):
        """
        Compute weights for WeightedRandomSampler.
        """
        class_counts = {}

        for slide_id in self.slide_ids:
            label = self.get_label(slide_id)
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        self.weights = []
        for slide_id in self.slide_ids:
            label = self.get_label(slide_id)
            self.weights.append(class_weights[label])

    def _load_features_into_memory(self):
        """
        Loads features for all experts and slides into memory.
        Uses a dictionary structure for efficient access:
        - Keys are slide IDs.
        - Values are dictionaries where:
            - Keys are expert names.
            - Values are tensors containing features for that expert on that slide.
        """
        features_in_memory = defaultdict(lambda: defaultdict(list))
        for slide_id in tqdm(self.slide_ids, desc="Loading features into memory"):
            for expert in self.experts:
                h5_path = os.path.join(
                    self.feat_folder, f"{slide_id}/{self.magnification}x_features.h5"
                )
                with h5py.File(h5_path, "r") as h5f:
                    dataset_name = f"{expert}_features"
                    features_in_memory[slide_id][expert] = torch.from_numpy(
                        h5f[dataset_name][()]
                    )
        return features_in_memory

    def calculate_class_counts(self):
        """
        Calculates the number of samples for each class.
        """
        self.class_counts = self.csv[self.label_column].value_counts().to_dict()

    def __len__(self):
        return len(self.slide_ids)

    def init_labels(self):
        unique_labels = self.csv[self.label_column].unique()
        self.labels = list(unique_labels)
        self.total_labels = len(self.labels)

    def get_label(self, slide_id):
        return self.csv.loc[self.csv["uuid"] == slide_id, self.label_column].values[0]

    def get_clinical_data(self, slide_id):
        row = self.csv.loc[self.csv["uuid"] == slide_id]
        # Define default values
        default_sex = -1
        default_age = -1
        default_race_value = -1

        sex = row["Sex"].fillna(default_sex).values[0]
        age = row["Age"].fillna(default_age).values[0]
        race_american_indian = int(
            row["Race_American Indian or Alaska Native"]
            .fillna(default_race_value)
            .values[0]
        )
        race_asian = int(row["Race_Asian"].fillna(default_race_value).values[0])
        race_black = int(
            row["Race_Black or African American"].fillna(default_race_value).values[0]
        )
        race_white = int(row["Race_White"].fillna(default_race_value).values[0])

        return [sex, age, race_american_indian, race_asian, race_black, race_white]

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]

        embeddings_list = []
        mask_list = []

        for expert in self.experts:
            embeddings = self.features_in_memory[slide_id][expert]

            embeddings_for_patches = []
            padded_indices = []

            if self.random_selection and self.n_patches is not None:
                selected_indices = sample(
                    range(len(embeddings)), min(len(embeddings), self.n_patches)
                )
                embeddings = embeddings[selected_indices]
            elif self.n_patches is not None:
                embeddings = embeddings[: self.n_patches]

            for i, embedding in enumerate(embeddings):
                if i < len(embeddings):
                    embeddings_for_patches.append(embedding)
                else:
                    zero_tensor = torch.zeros_like(embeddings[0])
                    embeddings_for_patches.append(zero_tensor)
                    padded_indices.append(i)

            target_size = EMBEDDING_SIZES[str(expert).lower()]
            additional_patch_count = self.n_patches - len(embeddings_for_patches)
            if additional_patch_count > 0:
                additional_patches = [
                    torch.zeros(target_size) for _ in range(additional_patch_count)
                ]
                embeddings_for_patches.extend(additional_patches)
                padded_indices.extend(
                    range(
                        len(embeddings_for_patches) - additional_patch_count,
                        len(embeddings_for_patches),
                    )
                )

            embeddings_list.append(torch.stack(embeddings_for_patches))

            mask = torch.zeros(len(embeddings_for_patches), dtype=torch.bool)
            mask[padded_indices] = True
            mask = mask.reshape(-1, 1)
            mask_list.append(mask)

        label = self.get_label(slide_id)

        if self.with_clinical:
            clinical_data = self.get_clinical_data(slide_id)
            clinical_tensor = torch.tensor(clinical_data, dtype=torch.float32)
            new_embeddings_list = []
            for embeddings_tensor in embeddings_list:
                embeddings_with_clinical = torch.cat(
                    [
                        embeddings_tensor,
                        clinical_tensor.repeat(embeddings_tensor.size(0), 1),
                    ],
                    dim=1,
                )
                new_embeddings_list.append(embeddings_with_clinical)
            embeddings_list = new_embeddings_list

        return slide_id, embeddings_list, mask_list, label

    def plot_label_distribution(self):
        self.csv[self.label_column].value_counts().plot(kind="bar")


class H5SingleFeatureMultiExpertDataset(Dataset):
    """
    This dataset class loads features for specific experts at a given magnification level
    from h5 files.

    In the training mode, it randomly or sequentially selects one feature at a time from the available
    patches per slide, according to the 'random_selection' setting. This allows the model to see
    different subsets of patches in different epochs, depending on the randomness.

    In the validation mode (`is_validation=True`), it operates differently:
    - It samples 'n_patches' features at once for each expert for a single whole slide image.
    - If the number of available patches is fewer than 'n_patches', it pads the remaining patches with zeros.
    - Returns a mask indicating which patches are real and which are padding.

    Parameters:
        csv_path (str): Path to the csv file containing metadata about slides.
        feat_folder (str): Directory containing feature embeddings from the experts for each slide.
        magnification (int): Magnification level used for the features.
        experts (List[str]): List of experts (model names) whose features are to be used.
        n_patches (int): Defines how many total patches per slide to look at.
        label_column (str): Name of the column in the csv that contains the labels.
        random_selection (bool): If True, patches are randomly selected.
        wsi_type (str): Type of slide preparation, e.g., 'permanent' or 'frozen'. May be ignored, if the column is not present in the csv.
        is_validation (bool): Flag to toggle the validation mode. Changes behavior for patch selection and output format.
    """

    def __init__(
        self,
        csv_path: str,
        feat_folder: str,
        n_patches: int,
        magnification: int,
        experts: List[str],
        random_selection: bool = False,
        wsi_type: str = "frozen",
        is_validation: bool = False,
    ):
        self.csv_path = csv_path
        self.feat_folder = feat_folder
        self.mag = magnification
        self.experts = [expert.upper() for expert in experts]
        self.n_patches = n_patches
        self.label_column = "label"
        self.wsi_type = wsi_type.lower()
        self.random_selection = random_selection
        self.is_validation = is_validation

        self.csv = pd.read_csv(csv_path)
        if "wsi_type" in self.csv.columns:
            self.csv = self.csv[self.csv["wsi_type"] == self.wsi_type]
        else:
            print("Warning: No 'wsi_type' column found in csv!")
        self.csv = self.csv[self.csv["uuid"].isin(os.listdir(self.feat_folder))]
        self.total_slides = len(self.csv["uuid"].unique())
        self.slide_ids = self.csv["uuid"].unique()

        self.labels = {
            uuid: row[self.label_column]
            for uuid, row in self.csv.set_index("uuid").iterrows()
        }
        self.indexed_features = self._index_features()
        self.init_labels()
        if not self.is_validation:
            self.indices_cache = self._load_indices()

    def _load_indices(self):
        indices_cache = {}
        for slide_id in tqdm(self.slide_ids, desc="Caching indices"):
            indices_cache[slide_id] = {}
            for expert in self.experts:
                h5_path = os.path.join(
                    self.feat_folder, f"{slide_id}/{self.mag}x_features.h5"
                )
                with h5py.File(h5_path, "r") as h5f:
                    dataset_name = f"{expert}_indices"
                    if dataset_name in h5f:
                        indices_cache[slide_id][expert] = h5f[dataset_name][()]
        return indices_cache

    def _index_features(self):
        indexed_features = []
        for slide_id in tqdm(self.slide_ids, desc="Indexing features"):
            num_patches = min(
                [
                    h5py.File(
                        os.path.join(
                            self.feat_folder, f"{slide_id}/{self.mag}x_features.h5"
                        ),
                        "r",
                    )[f"{expert}_features"].shape[0]
                    for expert in self.experts
                ]
            )

            if self.random_selection and not self.is_validation:
                selected_indices = sample(
                    range(num_patches), min(num_patches, self.n_patches)
                )
            else:
                selected_indices = list(range(min(num_patches, self.n_patches)))

            for idx in selected_indices:
                for expert in self.experts:
                    indexed_features.append((slide_id, expert, idx))

        return indexed_features

    def init_labels(self):
        unique_labels = self.csv[self.label_column].unique()
        self.labels = list(unique_labels)
        self.total_labels = len(self.labels)

    def get_label(self, slide_id):
        return self.csv.loc[self.csv["uuid"] == slide_id, self.label_column].values[0]

    def __len__(self):
        if self.is_validation:
            return len(self.slide_ids)
        return len(self.indexed_features)

    def __getitem__(self, idx):
        if self.is_validation:
            slide_id = self.slide_ids[idx]
            features = []
            masks = []
            for expert in self.experts:
                expert_features = []
                mask = []
                h5_path = os.path.join(
                    self.feat_folder, f"{slide_id}/{self.mag}x_features.h5"
                )
                with h5py.File(h5_path, "r") as h5f:
                    dataset_name = f"{expert}_features"
                    num_available_patches = h5f[dataset_name].shape[0]
                    num_patches_to_load = min(self.n_patches, num_available_patches)

                    for i in range(num_patches_to_load):
                        embeddings = torch.from_numpy(h5f[dataset_name][i])
                        expert_features.append(embeddings)
                        mask.append(False)

                    # pad remaining patches if needed
                    if num_patches_to_load < self.n_patches:
                        num_missing_patches = self.n_patches - num_patches_to_load
                        pad_shape = h5f[dataset_name][0].shape
                        padding = torch.zeros(pad_shape)
                        for _ in range(num_missing_patches):
                            expert_features.append(padding)
                            mask.append(True)

                features.append(torch.stack(expert_features))
                masks.append(torch.tensor(mask).unsqueeze(-1))

            label = self.get_label(slide_id)
            return features, masks, label
        else:
            slide_id, _, feature_idx = self.indexed_features[idx]
            features = []
            for expert in self.experts:
                h5_path = os.path.join(
                    self.feat_folder, f"{slide_id}/{self.mag}x_features.h5"
                )
                with h5py.File(h5_path, "r") as h5f:
                    dataset_name = f"{expert}_features"
                    if feature_idx < h5f[dataset_name].shape[0]:
                        embeddings = torch.from_numpy(h5f[dataset_name][feature_idx])
                    else:
                        pad_shape = h5f[dataset_name][0].shape
                        embeddings = torch.zeros(pad_shape)
                    features.append(embeddings)
            label = self.get_label(slide_id)
            return features, label


class H5SingleFeatureMultiExpertDatasetInMemory(Dataset):
    """
    This dataset class loads features for specific experts at a given magnification level
    from h5 files, but stores all the features in memory.
    This significantly speeds up data loading, especially for large batch sizes.

    !! This requires a lot of memory, so use with caution !!

    In the training mode, it randomly or sequentially selects one feature at a time from the available
    patches per slide, according to the 'random_selection' setting. This allows the model to see
    different subsets of patches in different epochs, depending on the randomness.

    In the validation mode (`is_validation=True`), it operates differently:
    - It samples 'n_patches' features at once for each expert for a single whole slide image.
    - If the number of available patches is fewer than 'n_patches', it pads the remaining patches with zeros.
    - Returns a mask indicating which patches are real and which are padding.

    Parameters:
        csv_path (str): Path to the csv file containing metadata about slides.
        feat_folder (str): Directory containing feature embeddings from the experts for each slide.
        magnification (int): Magnification level used for the features.
        experts (List[str]): List of experts (model names) whose features are to be used.
        n_patches (int): Defines how many total patches per slide to look at.
        label_column (str): Name of the column in the csv that contains the labels.
        random_selection (bool): If True, patches are randomly selected.
        wsi_type (str): Type of slide preparation, e.g., 'permanent' or 'frozen'. May be ignored, if the column is not present in the csv.
        is_validation (bool): Flag to toggle the validation mode. Changes behavior for patch selection and output format.
    """

    def __init__(
        self,
        csv_path: str,
        feat_folder: str,
        n_patches: int,
        magnification: int,
        experts: List[str],
        random_selection: bool = False,
        wsi_type: str = "frozen",
        is_validation: bool = False,
        limit: Optional[int] = None,
    ):
        self.csv_path = csv_path
        self.feat_folder = feat_folder
        self.mag = magnification
        self.experts = [expert.upper() for expert in experts]
        self.n_patches = n_patches
        self.label_column = "label"
        self.wsi_type = wsi_type.lower()
        self.random_selection = random_selection
        self.is_validation = is_validation

        self.csv = pd.read_csv(csv_path)
        if "wsi_type" in self.csv.columns:
            self.csv = self.csv[self.csv["wsi_type"] == self.wsi_type]
        else:
            print("Warning: No 'wsi_type' column found in csv!")
        self.csv = self.csv[self.csv["uuid"].isin(os.listdir(self.feat_folder))]
        self.total_slides = len(self.csv["uuid"].unique())
        self.slide_ids = self.csv["uuid"].unique()
        if limit:
            self.slide_ids = self.slide_ids[:limit]

        self.labels = {
            uuid: row[self.label_column]
            for uuid, row in self.csv.set_index("uuid").iterrows()
        }

        self.features_in_memory = self._load_features_into_memory()
        self.init_labels()
        self.indexed_features = self._index_features()
        self.compute_weights()

    def compute_weights(self):
        """
        Computes weights for each item in the dataset based on the class distribution.
        These weights can then be used in the WeightedRandomSampler.

        ```python
        from torch.utils.data import WeightedRandomSampler, DataLoader
        sampler = WeightedRandomSampler(dataset.weights, num_samples=len(dataset.weights), replacement=True)
        dl = DataLoader(dataset=dataset, batch_size=2048, sampler=sampler)
        ```
        """
        class_counts = {}
        if not self.is_validation:
            for i in range(len(self.indexed_features)):
                slide_id, _ = self.indexed_features[i]
                label = self.get_label(slide_id)
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1
        else:
            for slide_id in self.slide_ids:
                label = self.get_label(slide_id)
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1

        self.class_counts = class_counts
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

        self.weights = []
        if not self.is_validation:
            for i in range(len(self.indexed_features)):
                slide_id, _ = self.indexed_features[i]
                label = self.get_label(slide_id)
                self.weights.append(class_weights[label])
        else:
            for slide_id in self.slide_ids:
                label = self.get_label(slide_id)
                self.weights.append(class_weights[label])

    def _load_features_into_memory(self):
        features_in_memory = defaultdict(lambda: defaultdict(list))
        for slide_id in tqdm(self.slide_ids, desc="Loading features into memory"):
            for expert in self.experts:
                h5_path = os.path.join(
                    self.feat_folder, f"{slide_id}/{self.mag}x_features.h5"
                )
                with h5py.File(h5_path, "r") as h5f:
                    dataset_name = f"{expert}_features"
                    features_in_memory[slide_id][expert] = torch.from_numpy(
                        h5f[dataset_name][()]
                    )
        return features_in_memory

    def _index_features(self):
        indexed_features = []
        for slide_id in tqdm(self.slide_ids, desc="Indexing features"):
            num_patches = min(
                len(self.features_in_memory[slide_id][expert])
                for expert in self.experts
            )
            indices = list(range(num_patches))
            if self.random_selection and not self.is_validation:
                indices = sample(indices, min(num_patches, self.n_patches))
            else:
                indices = indices[: self.n_patches]

            for idx in indices:
                indexed_features.append((slide_id, idx))

        return indexed_features

    def init_labels(self):
        unique_labels = self.csv[self.label_column].unique()
        self.labels = list(unique_labels)
        self.total_labels = len(self.labels)

    def get_label(self, slide_id):
        return self.csv.loc[self.csv["uuid"] == slide_id, self.label_column].values[0]

    def __len__(self):
        if self.is_validation:
            return len(self.slide_ids)
        return len(self.indexed_features)

    def __getitem__(self, idx):
        if self.is_validation:
            slide_id = self.slide_ids[idx]
            features = []
            masks = []
            for expert in self.experts:
                expert_features = []
                mask = []
                num_available_patches = len(self.features_in_memory[slide_id][expert])
                num_patches_to_load = min(self.n_patches, num_available_patches)

                for i in range(num_patches_to_load):
                    embeddings = self.features_in_memory[slide_id][expert][i]
                    expert_features.append(embeddings)
                    mask.append(False)

                if num_patches_to_load < self.n_patches:
                    num_missing_patches = self.n_patches - num_patches_to_load
                    pad_shape = self.features_in_memory[slide_id][expert][0].shape
                    padding = torch.zeros(pad_shape)
                    for _ in range(num_missing_patches):
                        expert_features.append(padding)
                        mask.append(True)

                features.append(torch.stack(expert_features))
                masks.append(torch.tensor(mask).unsqueeze(-1))

            label = self.get_label(slide_id)
            return slide_id, features, masks, label
        else:
            # when we get a single feature per expert at a time
            slide_id = self.indexed_features[idx][0]
            feature_idx = self.indexed_features[idx][1]

            features = []
            for expert in self.experts:
                embeddings = self.features_in_memory[slide_id][expert][feature_idx]
                features.append(embeddings)

            label = self.get_label(slide_id)
            return slide_id, features, label


class H5SingleFeatureMultiScaleExpertDatasetInMemory(Dataset):
    """
    This dataset class loads features for specific experts at a their given magnification level
    from h5 files, but stores all the features in memory.
    This significantly speeds up data loading, especially for large batch sizes.

    !! This requires a lot of memory, so use with caution !!

    In the training mode, it randomly or sequentially selects one feature at a time from the available
    patches per slide, according to the 'random_selection' setting. This allows the model to see
    different subsets of patches in different epochs, depending on the randomness.

    In the validation mode (`is_validation=True`), it operates differently:
    - It samples 'n_patches' features at once for each expert for a single whole slide image.
    - If the number of available patches is fewer than 'n_patches', it pads the remaining patches with zeros.
    - Returns a mask indicating which patches are real and which are padding.

    Parameters:
        csv_path (str): Path to the csv file containing metadata about slides.
        feat_folder (str): Directory containing feature embeddings from the experts for each slide.
        magnifications (List[int]): Magnification levels used for the features.
        experts (List[str]): List of experts (model names) whose features are to be used.
        n_patches (int): Defines how many total patches per slide to look at.
        label_column (str): Name of the column in the csv that contains the labels.
        wsi_type (str): Type of slide preparation, e.g., 'permanent' or 'frozen'. May be ignored, if the column is not present in the csv.
    """

    def __init__(
        self,
        csv_path: str,
        feat_folder: str,
        n_patches: int,
        magnifications: List[int],
        experts: List[str],
        wsi_type: str = "permanent",
        limit: Optional[int] = None,
    ):
        self.csv_path = csv_path
        self.feat_folder = feat_folder
        self.magnifications = magnifications
        self.experts = [expert.upper() for expert in experts]
        self.n_patches = n_patches
        self.label_column = "label"
        self.wsi_type = wsi_type.lower()

        assert len(self.experts) == len(
            self.magnifications
        ), "Experts and magnifications lists must have the same length."

        self.csv = pd.read_csv(csv_path)
        if "wsi_type" in self.csv.columns:
            self.csv = self.csv[self.csv["wsi_type"] == self.wsi_type]
        else:
            print("Warning: No 'wsi_type' column found in csv!")
        self.csv = self.csv[self.csv["uuid"].isin(os.listdir(self.feat_folder))]
        self.total_slides = len(self.csv["uuid"].unique())
        self.slide_ids = self.csv["uuid"].unique()
        if limit:
            self.slide_ids = self.slide_ids[:limit]

        self.labels = {
            uuid: row[self.label_column]
            for uuid, row in self.csv.set_index("uuid").iterrows()
        }

        self.features_in_memory = self._load_features_into_memory()
        self.init_labels()
        self.indexed_features = self._index_features()
        self.compute_weights()

    def compute_weights(self):
        """
        Computes weights for each item in the dataset based on the class distribution.
        These weights can then be used in the WeightedRandomSampler.

        ```python
        from torch.utils.data import WeightedRandomSampler, DataLoader
        sampler = WeightedRandomSampler(dataset.weights, num_samples=len(dataset.weights), replacement=True)
        dl = DataLoader(dataset=dataset, batch_size=2048, sampler=sampler)
        ```
        """
        class_counts = {}
        for i in range(len(self.indexed_features)):
            slide_id, _ = self.indexed_features[i]
            label = self.get_label(slide_id)
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        self.class_counts = class_counts
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

        self.weights = []
        for i in range(len(self.indexed_features)):
            slide_id, _ = self.indexed_features[i]
            label = self.get_label(slide_id)
            self.weights.append(class_weights[label])

    def _load_features_into_memory(self):
        features_in_memory = defaultdict(lambda: defaultdict(list))
        for slide_id in tqdm(self.slide_ids, desc="Loading features into memory"):
            for expert, magnification in zip(self.experts, self.magnifications):
                h5_path = os.path.join(
                    self.feat_folder,
                    f"{slide_id}/{magnification}x_features.h5",
                )
                with h5py.File(h5_path, "r") as h5f:
                    dataset_name = f"{expert}_features"
                    features_in_memory[slide_id][(expert, magnification)] = (
                        torch.from_numpy(h5f[dataset_name][()])
                    )
        return features_in_memory

    def _index_features(self):
        indexed_features = []
        for slide_id in tqdm(self.slide_ids, desc="Indexing features"):
            num_patches = min(
                len(self.features_in_memory[slide_id][(expert, mag)])
                for expert, mag in zip(self.experts, self.magnifications)
            )
            indices = list(range(num_patches))
            indices = sample(indices, min(num_patches, self.n_patches))

            for idx in indices:
                indexed_features.append((slide_id, idx))

        return indexed_features

    def init_labels(self):
        unique_labels = self.csv[self.label_column].unique()
        self.labels = list(unique_labels)
        self.total_labels = len(self.labels)

    def get_label(self, slide_id):
        return self.csv.loc[self.csv["uuid"] == slide_id, self.label_column].values[0]

    def __len__(self):
        return len(self.indexed_features)

    def __getitem__(self, idx):
        # when we get a single feature per expert at a time
        slide_id = self.indexed_features[idx][0]
        feature_idx = self.indexed_features[idx][1]

        features = []
        for expert, mag in zip(self.experts, self.magnifications):
            embeddings = self.features_in_memory[slide_id][(expert, mag)][feature_idx]
            features.append(embeddings)

        label = self.get_label(slide_id)
        return slide_id, features, label


class H5MultiScaleDatasetInMemory(Dataset):
    """
    This dataset class loads and processes features from multiple experts stored in h5 files,
    keeping all the features in memory for fast access.

    !! This requires a lot of memory, so use with caution !!

    Each item returned by this dataset includes:
        - A list of tensors, where each tensor represents the embeddings for a slide's patches
          as processed by a different expert model at a different magnification.
        - A list of tensors, where each tensor is a boolean mask indicating whether the
          corresponding patch was padded.
        - The label for the whole slide

    Parameters:
        csv_path (str): Path to the csv file containing metadata about slides.
        feat_folder (str): Directory containing feature embeddings from the experts for each slide.
        experts (List[str]): List of experts (model names) whose features are to be used.
        magnifications (List[int]): List of magnifications to load features from.
        n_patches (int, optional): Number of patches to include per slide. If None, all patches are used.
        label_column (str): Name of the column in the csv that contains the labels.
        wsi_type (str): Type of slide preparation, e.g., 'permanent' or 'frozen'. May be ignored
                        if the column is not present in the csv.
        verbose (bool): If True, prints detailed logs and analyses of the dataset during initialization.

    TODO:
        - implement logic for get_metadata (see create_features script) => metadata is stored in [mag]x_features.h5, dataset "metadata"
    """

    def __init__(
        self,
        csv_path: str,
        feat_folder: str,
        experts: List[str],
        magnifications: List[int],
        n_patches=None,
        label_column: str = "label",
        wsi_type: str = "permanent",
        verbose: bool = False,
        get_metadata=False,
        random_selection=False,
        with_clinical: bool = False,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.feat_folder = feat_folder
        self.experts = [expert.upper() for expert in experts]
        self.magnifications = magnifications
        self.n_patches = n_patches
        self.label_column = label_column
        self.wsi_type = str(wsi_type).lower()
        self.verbose = verbose
        self.get_metadata = get_metadata
        self.random_selection = random_selection
        self.with_clinical = with_clinical
        # for now hardcoded
        self.clinical_columns = [
            "Age",
            "Sex",
            "Race_American Indian or Alaska Native",
            "Race_Asian",
            "Race_Black or African American",
            "Race_White",
        ]
        assert len(self.experts) == len(
            self.magnifications
        ), "Experts and magnifications lists must have the same length."

        self.csv = pd.read_csv(csv_path)
        if "wsi_type" in self.csv.columns:
            self.csv = self.csv[self.csv["wsi_type"] == self.wsi_type]
        else:
            print("Warning: No 'wsi_type' column found in csv!")
        self.csv = self.csv[self.csv["uuid"].isin(os.listdir(self.feat_folder))]

        if with_clinical:
            # verify that all columns are in self.csv
            for col in self.clinical_columns:
                if col not in self.csv.columns:
                    raise ValueError(
                        f"Clinical data column {col} was not found in csv: {csv_path}"
                    )

        self.total_slides = len(self.csv["uuid"].unique())
        self.slide_ids = self.csv["uuid"].unique()

        success_ids = load_success_ids(self.feat_folder)
        self.slide_ids = [x for x in self.slide_ids if x in success_ids]

        self.init_labels()
        self.calculate_class_counts()
        self.label_counts = Counter(self.csv[self.label_column])
        self.features_in_memory = self._load_features_into_memory()
        self.compute_weights()

    def compute_weights(self):
        """
        Compute weights for WeightedRandomSampler.
        """
        class_counts = {}

        for slide_id in self.slide_ids:
            label = self.get_label(slide_id)
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        self.weights = []
        for slide_id in self.slide_ids:
            label = self.get_label(slide_id)
            self.weights.append(class_weights[label])

    def _load_features_into_memory(self):
        """
        Loads features for all experts and slides into memory.
        Uses a dictionary structure for efficient access:
        - Keys are slide IDs.
        - Values are dictionaries where:
            - Keys are tuples of (expert, magnification).
            - Values are tensors containing features for that expert and magnification on that slide.
        """
        features_in_memory = defaultdict(lambda: defaultdict(list))
        for slide_id in tqdm(self.slide_ids, desc="Loading features into memory"):
            for expert, magnification in zip(self.experts, self.magnifications):
                h5_path = os.path.join(
                    self.feat_folder,
                    f"{slide_id}/{magnification}x_features.h5",
                )
                with h5py.File(h5_path, "r") as h5f:
                    dataset_name = f"{expert}_features"
                    features_in_memory[slide_id][(expert, magnification)] = (
                        torch.from_numpy(h5f[dataset_name][()])
                    )
        return features_in_memory

    def calculate_class_counts(self):
        """
        Calculates the number of samples for each class.
        """
        self.class_counts = self.csv[self.label_column].value_counts().to_dict()

    def __len__(self):
        return len(self.slide_ids)

    def init_labels(self):
        unique_labels = self.csv[self.label_column].unique()
        self.labels = list(unique_labels)
        self.total_labels = len(self.labels)

    def get_label(self, slide_id):
        return self.csv.loc[self.csv["uuid"] == slide_id, self.label_column].values[0]

    def get_clinical_data(self, slide_id):
        row = self.csv.loc[self.csv["uuid"] == slide_id]
        # Define default values
        default_sex = -1
        default_age = -1
        default_race_value = -1

        sex = row["Sex"].fillna(default_sex).values[0]
        age = row["Age"].fillna(default_age).values[0]
        race_american_indian = int(
            row["Race_American Indian or Alaska Native"]
            .fillna(default_race_value)
            .values[0]
        )
        race_asian = int(row["Race_Asian"].fillna(default_race_value).values[0])
        race_black = int(
            row["Race_Black or African American"].fillna(default_race_value).values[0]
        )
        race_white = int(row["Race_White"].fillna(default_race_value).values[0])

        return [sex, age, race_american_indian, race_asian, race_black, race_white]

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]

        embeddings_list = []
        mask_list = []

        # select the indices of the features we will pick
        if self.random_selection and self.n_patches is not None:
            selected_indices = sample(
                range(
                    len(
                        self.features_in_memory[slide_id][
                            (self.experts[0], self.magnifications[0])
                        ]
                    )
                ),
                min(
                    len(
                        self.features_in_memory[slide_id][
                            (self.experts[0], self.magnifications[0])
                        ]
                    ),
                    self.n_patches,
                ),
            )
        elif self.n_patches is not None:
            selected_indices = list(range(self.n_patches))

        # for each expert/magnification pair, load the features belonging to the same patch regions
        for expert, magnification in zip(self.experts, self.magnifications):
            embeddings = self.features_in_memory[slide_id][(expert, magnification)]
            embeddings_for_patches = []
            padded_indices = []

            for i in selected_indices:
                if i < len(embeddings):
                    embeddings_for_patches.append(embeddings[i])
                else:
                    zero_tensor = torch.zeros_like(embeddings[0])
                    embeddings_for_patches.append(zero_tensor)
                    padded_indices.append(i)

            target_size = EMBEDDING_SIZES[str(expert).lower()]
            additional_patch_count = self.n_patches - len(embeddings_for_patches)
            if additional_patch_count > 0:
                additional_patches = [
                    torch.zeros(target_size) for _ in range(additional_patch_count)
                ]
                embeddings_for_patches.extend(additional_patches)
                padded_indices.extend(
                    range(
                        len(embeddings_for_patches) - additional_patch_count,
                        len(embeddings_for_patches),
                    )
                )

            embeddings_list.append(torch.stack(embeddings_for_patches))

            # create a boolean mask based on the padded indices
            mask = torch.zeros(len(embeddings_for_patches), dtype=torch.bool)
            mask[padded_indices] = True
            mask = mask.reshape(-1, 1)
            mask_list.append(mask)

        label = self.get_label(slide_id)

        if self.with_clinical:
            clinical_data = self.get_clinical_data(slide_id)
            clinical_tensor = torch.tensor(clinical_data, dtype=torch.float32)
            new_embeddings_list = []
            for embeddings_tensor in embeddings_list:
                embeddings_with_clinical = torch.cat(
                    [
                        embeddings_tensor,
                        clinical_tensor.repeat(embeddings_tensor.size(0), 1),
                    ],
                    dim=1,
                )
                new_embeddings_list.append(embeddings_with_clinical)
            embeddings_list = new_embeddings_list

        return slide_id, embeddings_list, mask_list, label

    def plot_label_distribution(self):
        self.csv[self.label_column].value_counts().plot(kind="bar")


class H5PatchDatasetInMemory(Dataset):
    """
    Proof-of-concept dataset for end-to-end training.

    ## Note

    This has not been extensively tested yet and has hardcoded features.
    You should not really use this yet!
    """

    def __init__(
        self,
        csv_path: str,
        folder: str,
        magnifications: List[int],
        transform: Optional[Callable] = get_transforms(),
        n_patches: int = 250,
        verbose: bool = False,
        get_metadata=False,
        random_selection=False,
        limit: Optional[int] = None,
    ):
        super().__init__()
        assert (
            len(magnifications) > 0
        ), "At least one magnification level must be provided."
        self.csv = pd.read_csv(csv_path)
        self.folder = folder
        self.magnifications = magnifications
        self.transform = transform
        self.verbose = verbose
        self.n_patches = n_patches
        self.get_metadata = get_metadata
        self.random_selection = random_selection
        self.slide_ids = self.csv["uuid"].unique()
        success_ids = load_success_ids(self.folder)
        self.slide_ids = [x for x in self.slide_ids if x in success_ids]
        if limit:
            self.slide_ids = self.slide_ids[:limit]
        self.labels = []
        self.patches = []
        self.load_patches()
        self.compute_weights()

    def load_patches(self):
        """
        Load n_patches into memory.
        """
        for slide_id in tqdm(self.slide_ids, desc="Prefetch patches"):
            # TODO: adjust `_features.h5` once we renamed it on the storage server
            file = f"{self.folder}/{slide_id}_features.h5"
            with h5py.File(file, "r") as h5f:
                n_patches = min(self.n_patches, len(h5f[str(self.magnifications[0])]))

                # select random indices
                if self.random_selection:
                    indices = sample(range(n_patches), n_patches)
                else:
                    indices = list(range(n_patches))

                for i in indices:
                    imgs = [
                        Image.fromarray(h5f[str(mag)][i]) for mag in self.magnifications
                    ]
                    self.patches.append((imgs, slide_id))
                    self.labels.append(self.get_label(slide_id))

    def __len__(self):
        return len(self.patches)

    def get_label(self, slide_id):
        return self.csv.loc[self.csv["uuid"] == slide_id, "label"].values[0]

    def compute_weights(self):
        """
        Compute weights for WeightedRandomSampler.
        """
        class_counts = {}
        for label in self.labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        self.weights = [class_weights[label] for label in self.labels]

    def __getitem__(self, idx):
        imgs, slide_id = self.patches[idx]
        imgs = self.transform(imgs)
        label = self.get_label(slide_id)
        return slide_id, imgs, label
