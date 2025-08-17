"""
# Feature Extraction

Creates features based on the patches extracted using create_tiles.py.
The user can select a set of foundational models to extract features from the patches.
Currently, we support the following models:

- lunit
- resnet50
- uni
- swin224
- phikon
- ctrans
- chief
- plip
- gigapath
- cigar
- virchow
- virchow2
"""

import argparse
import glob
import math
import os
import pprint
import sqlite3
import time
import openslide
import traceback
from functools import wraps
from typing import Dict, List, Set

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, sampler, Subset

from models.library import get_model, parse_model_type
from utils.transforms import TorchBatchMacenkoNormalizer
from utils.transforms import get_transforms_timm as get_transforms
from  torchvision.transforms.functional import pil_to_tensor, to_pil_image
import stat
TQDM_MIN_INTERVAL=60


# pylint: disable=line-too-long
def parse_args():
    parser = argparse.ArgumentParser(
        usage="""python create_features_from_patches.py
          --patch_folder PATH_TO_PATCHES
          --feat_folder PATH_TO_features
          [--models MODEL_TYPES]""",
    )
    parser.add_argument(
        "--patch_folder",
        type=str,
        help="Root patch folder. Patches are expected in <patch_folder>/<slide_id>.csv and <patch_folder>/<slide_id>/<idx>_<mag>.png.",
    )
    parser.add_argument(
        "--wsi_folder",
        type=str,
        help="Root wsi folder. Patches are expected in <wsi_folder>/<slide_id>.svs.",
    )
    parser.add_argument(
        "--feat_folder",
        type=str,
        help="Root folder, under which the features will be stored: <feature_folder>/<slide_id>/",
    )
    parser.add_argument(
        "--wsi_col",
        type=str,
        default='filepath',
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default='ID',
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="CSV file containing the slide IDs and wsi path to process.",
    )
    parser.add_argument(
        "--read_patch_size",
        type=int,
        default=0,
        help="Read patch size for processing the patches. Default: 0.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing the patches. Default: 64.",
    )
    parser.add_argument(
        "--models",
        type=parse_model_type,
        default=[
            "ctrans",
            "lunit",
            "resnet50",
            "uni",
            "swin224",
            "phikon",
            "chief",
            "plip",
            "gigapath",
            "cigar",
            "virchow",
            "virchow2",
        ],
        help="Comma-separated list of models to use (e.g., 'lunit,resnet50,uni,swin224,phikon').",
    )
    
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of workers to use for processing patches in parallel(default 8)",
    )
    parser.add_argument(
        "--n_parts",
        type=int,
        default=1,
        help="The number of parts to split the items into (default 1)",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=0,
        help="The part of the total items to process (default 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to use for processing (default 'cuda')",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="The huggingface token to use for downloading models (default None)",
    )
    parser.add_argument(
        "--target_mag",
        type=int,
        default=40,
        help="The magnification power to output the tiles at",
    )
    parser.add_argument(
        "--stain_norm",
        action='store_true',
        help="Whether to apply stain normalization",
    )
    parser.add_argument(
        "--max_num_patches",
        type=int,
        default=None,
        help="""The maximum number of patches to process. If the number of patches is greater than this value, a random subset will be selected.
        If None, all patches will be processed.""")
    ## Added options for reading tiff files
    parser.add_argument(
        "--img_format",
        type=str,
        default="wsi",
        choices=["wsi", "tiff"],
        help="The format of the input images (default wsi). If tiff, the script will use the magnification from the argument.")
    parser.add_argument(
        "--tiff_mag",
        type=float,
        default=40,
        help="The magnification of the tiff images (default 40, for Phillips Scanners).")
    parser.add_argument(
        "--wsi_mag",
        type=int,
        default=-1,
        help="The magnification of the wsi images (default -1).")
    ##
    return parser.parse_args()

class WSIDataset(Dataset):
    def __init__(self,
        args,
        file_path,
        wsi,
        pretrained=False,
        custom_transforms=None,
        transform=True,
        custom_downsample=1,
        target_patch_size=-1,
        thumbnail_mask_path=None,
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained = pretrained
        self.transform = transform
        self.stain_norm = args.stain_norm
        self.wsi = wsi
        self.args = args
        self.target_patch_size = target_patch_size
        self.custom_downsample = custom_downsample
            
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms
        if self.stain_norm:
            import cv2
            
            ## find the thumbnail mask for stain normalization
            self.normalizer = TorchBatchMacenkoNormalizer(source_thumbnail=thumbnail_mask_path, source_thumbnail_mask=thumbnail_mask_path)

            # self.normalizer = torch_normalizer
        self.file_path = file_path
        ## initialize the patch parameters
        self.init_patch_params()

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        for name, value in self.metadata.items():
            print(f"{name}: {value}")
        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)
        print('stain_norm: ', self.stain_norm)
    def init_patch_params(self):
        '''
        Initialize the parameters for reading the patches
        '''
        ############################## 
        ## 1. read the patch size from the the metadata
        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.metadata = {}
            for name in f['metadata'].dtype.names:
                value = f['metadata'][name][0]  # Get the value for this field
                self.metadata[name] = value
            self.patch_size = f['metadata']['patch_size'][0]
            self.length = len(dset)
            # target_patch_size (usually not used)
            if self.target_patch_size > 0:
                self.target_patch_size = (self.target_patch_size, ) * 2
            elif self.custom_downsample > 1:
                self.target_patch_size = (self.patch_size // self.custom_downsample, ) * 2
            else:
                self.target_patch_size = None
        ############################## 
        ## 2. estimate the read patch size and the downsample rate at the target magnification
        if self.args.img_format == 'wsi':
            if self.args.wsi_mag > 0:
                highest_mag = self.args.wsi_mag
            else:
                highest_mag = float(self.wsi.properties["openslide.objective-power"])
            native_magnifications = {
                round(highest_mag / self.wsi.level_downsamples[level], 2): level
                for level in range(self.wsi.level_count)
            }
            if self.args.target_mag in native_magnifications:
                ## if the target mag is available, extract the patch at the target mag
                self.read_level = native_magnifications[self.args.target_mag]
                self.scale_factor = None
                self.read_patch_size = None
                print(f'target magnification ({self.args.target_mag}X) is available')

            else:
                ## if the target mag is not available, extract the patch at the nearest higher mag and downsample
                nearest_higher_mag = max(
                    [mag for mag in native_magnifications if mag > self.args.target_mag],
                    default=highest_mag,
                )
                nearest_higher_level = native_magnifications[nearest_higher_mag]
                scale_factor = nearest_higher_mag / self.args.target_mag
                self.read_level = nearest_higher_level
                self.scale_factor = scale_factor
                self.read_patch_size = (
                    round(self.patch_size * scale_factor),
                    round(self.patch_size * scale_factor),
                )
                print(f'target magnification ({self.args.target_mag}X) is not available. Will extract at {nearest_higher_mag}X and downsample')
        ## Added by SY: for tiff images, the magnification is fixed
        elif self.args.img_format == 'tiff':
            self.read_level = 0
            self.scale_factor = self.args.tiff_mag / self.args.target_mag
            if self.args.read_patch_size > 0:
                self.read_patch_size = (self.args.read_patch_size, self.args.read_patch_size)
            else:
                self.read_patch_size = (
                    round(self.patch_size * self.scale_factor),
                    round(self.patch_size * self.scale_factor),
                )
            print(f'For tiff format, use fixed magnification {self.args.tiff_mag}X and downsample to {self.args.target_mag}X')
            

    def extract_patch(self, coord):
        '''
        Extract the patch at the target magnification
        If the target magnification is not available, extract the patch at the nearest higher magnification and downsample
        '''
        if self.scale_factor is None:
            ## if the target mag is available, extract the patch at the target mag
            patch = self.wsi.read_region(coord, self.read_level, (self.patch_size, self.patch_size))
        else:
            ## if the target mag is not available, extract the patch at the nearest higher mag and downsample
            extract_size = self.read_patch_size
            ## extract the patch at the nearest higher mag
            patch = self.wsi.read_region(coord, self.read_level, extract_size)
            ## downsample the patch to the target mag
            patch = patch.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        patch = patch.convert('RGB')
        # patch = np.array(patch)
        return patch

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        
        # img = self.wsi.read_region(coord, self.has_target_mag(), (self.patch_size, self.patch_size)).convert('RGB')
        img = self.extract_patch(coord)

        if self.target_patch_size is not None:  ##disabled since it is already in the transforms
        #     img = img.resize(self.target_patch_size)
            raise NotImplementedError('target_patch_size should be already in the transforms')
        
        if self.stain_norm:
            try:
                # img, _, _ = self.normalizer.normalize(I=img, stains=False)
                img = pil_to_tensor(img)
                img = self.normalizer(img).type_as(img)

                img = to_pil_image(img)
                #
            except:
                pass
        else:
            # img = np.array(img, dtype=np.uint8)
            pass
        # img = Image.fromarray(img.astype(np.uint8))
            
        if self.transform:
            img = self.roi_transforms(img).unsqueeze(0)
            
        # return np.asarray(img), np.asarray(coord)
        return img, np.asarray(coord)

class H5Writer:
    """
    # H5Writer

    Efficient HDF5-based storage for feature extraction results.

    Provides an interface for writing feature tensors and associated metadata to HDF5 files,
    optimized for large-scale feature extraction from image datasets.

    ### Features:
    - Incremental writes with very low memory overhead
    - Automatic dataset creation and extension
    - Gzip compression for storage efficiency
    - Metadata preservation from source files
    - Direct handling of PyTorch tensors

    It supports multiple model outputs in a single file and ensures traceability
    by preserving original metadata.

    ### Usage:
        >>> writer = H5Writer('output_features.h5')
        >>> writer.push_features({'model1': tensor1, 'model2': tensor2})
        >>> writer.copy_metadata('source_data.h5')
        >>> writer.close()

    ### Note:
    This class is not explicitly thread-safe. For multi-threaded environments,
    implement external synchronization mechanisms.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.file = h5py.File(self.file_path, "a")

    def push_features(self, features: Dict[str, torch.Tensor]):
        for model_name, feature_tensor in features.items():
            dataset_name = f"{model_name}_features"
            feature_array = feature_tensor.cpu().numpy()

            if dataset_name not in self.file:
                self.file.create_dataset(
                    dataset_name,
                    data=feature_array,
                    maxshape=(None, feature_array.shape[1]),
                    chunks=True,
                    compression="gzip",
                    compression_opts=9,
                )
            else:
                dataset = self.file[dataset_name]
                current_size = dataset.shape[0]
                new_size = current_size + feature_array.shape[0]
                dataset.resize(new_size, axis=0)
                dataset[current_size:new_size] = feature_array

        self.file.flush()

    def copy_metadata(self, source_file):
        with h5py.File(source_file, "r") as src:
            if "metadata" in src:
                if "metadata" in self.file:
                    del self.file["metadata"]
                self.file.copy(src["metadata"], "metadata")

    def close(self):
        self.file.close()


def initialize_db(args):
    """Initialize the SQLite database to track processed slide IDs and the models that have been used on them."""
    if args.stain_norm:
        db_path = os.path.join(args.feat_folder, "success_stainnorm.db")
    else:
        db_path = os.path.join(args.feat_folder, "success.db")
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    ###
    st = os.stat(db_path)
    st_mode = st.st_mode    
    # os.chmod(db_path, 0o777)
    try:
        os.chmod(db_path, st_mode | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
    except Exception as e:
        pass
    ###
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS success (
                slide_id TEXT PRIMARY KEY,
                experts TEXT
            )
        """
        )
        conn.commit()
    finally:
        conn.close()


def load_success_data(args) -> pd.DataFrame:
    """
    Load the slide data from the SQLite database to track already processed slide IDs.

    ## Why a database?

    It allows us to run thousands of jobs in parallel on our infrastructure concurrently.
    (Over)writing to text files would craete concurrency problems.
    """
    # db_path = f"{args.feat_folder}/success.db"
    
    if args.stain_norm:
        db_path = os.path.join(args.feat_folder, "success_stainnorm.db")
    else:
        db_path = os.path.join(args.feat_folder, "success.db")
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM success", conn)
    conn.close()
    return df


def update_success(args, slide_id: str, models):
    """
    Update the database with the slide ID and the models that have successfully processed it.
    If the slide ID already exists, appends the new models to the existing list of models.
    """
    # db_path = f"{args.feat_folder}/success.db"
    if args.stain_norm:
        db_path = os.path.join(args.feat_folder, "success_stainnorm.db")
    else:
        db_path = os.path.join(args.feat_folder, "success.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT experts FROM success WHERE slide_id = ?", (slide_id,))
    result = cursor.fetchone()
    if result:
        existing_experts = set(result[0].split(","))
        updated_experts = existing_experts.union(models)
        cursor.execute(
            "UPDATE success SET experts = ? WHERE slide_id = ?",
            (",".join(updated_experts), slide_id),
        )
    else:
        cursor.execute(
            "INSERT INTO success (slide_id, experts) VALUES (?, ?)",
            (slide_id, ",".join(models)),
        )
    conn.commit()
    conn.close()


def setup_folders(args):
    """Create necessary directories for patches and features if they do not already exist."""
    os.makedirs(args.feat_folder, exist_ok=True)
    # os.makedirs(f"{args.feat_folder}/h5_files", exist_ok=True)
    # os.makedirs(f"{args.feat_folder}/pt_files", exist_ok=True)
    assert os.path.exists(
        args.patch_folder
    ), f"Patch folder {args.patch_folder} does not exist."


def load_available_patches(args) -> Set[str]:
    """
    Load the IDs of patches that are available for processing from a text file.

    ## Notes

    This assumes that the tile extraction script still uses the old method with the text file.
    """
    available_patches_txt = f"{args.patch_folder}/success.txt"
    available_patch_ids = set()
    if os.path.exists(available_patches_txt):
        with open(available_patches_txt, "r") as f:
            available_patch_ids = {line.strip() for line in f}
    return available_patch_ids


def load_all_h5s(args) -> List[str]:
    """Load all h5 files that list patch information for each slide ID."""
    available_ids = load_available_patches(args)
    files = glob.glob(f"{args.patch_folder}/coords/*.h5")
    files = sorted(files)
    files = [
        file
        for file in files
        if os.path.splitext(os.path.basename(file))[0] in available_ids
    ]
    return files

def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def retry(max_retries=3, delay=5, exceptions=(Exception,)):
    """Simple decorator to retry a function upon encountering specified exceptions."""

    def decorator_retry(func):
        @wraps(func)
        def wrapper_retry(*args, **kwargs):
            retries = max_retries
            while retries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries -= 1
                    if retries <= 0:
                        raise
                    print(f"Retry {func.__name__} due to {e}, {retries} retries left.")
                    time.sleep(delay)

        return wrapper_retry

    return decorator_retry


@retry(max_retries=15, delay=5, exceptions=(OSError,))
def load_models(args) -> Dict[str, nn.Module]:
    """
    Load the specified models into memory. Retry a couple of times if they fail.

    ## Why might this fail?

    Huggingface sometimes rate-limits downloads. If we run hundreds of jobs, we may get rate-limited.
    """
    models = {}
    
    for model in args.models:
        models[str(model)] = get_model(args, str(model)).to(args.device)
    return models


def load_patch(row, patch_folder: str, slide_id: str, transforms: nn.Module):
    """Load an individual patch image and apply transformations for model processing."""
    idx, mag = row["idx"], row["magnification"]
    patch_png = f"{patch_folder}/{slide_id}/{idx}_{mag}.png"
    if os.path.exists(patch_png):
        patch = Image.open(patch_png).convert("RGB")
        return idx, mag, transforms(patch)
    return None, None, None


def get_features(
    batch: torch.Tensor, models: Dict[str, nn.Module]
) -> Dict[str, torch.Tensor]:
    """Process a batch of images using the loaded models and return their features."""
    batch_features = {model_type: [] for model_type in models.keys()}
    with torch.no_grad():
        for model_type, model in models.items():
            batch_features[model_type] = model(batch).detach().cpu()
    return batch_features


def store_metadata(args, slide_id: str):
    """Store patch metadata such as index and magnification to an HDF5 file."""
    metadata = pd.read_csv(f"{args.patch_folder}/{slide_id}.csv")

    # drop column "uuid" if it exists
    if "slide_id" in metadata.columns:
        metadata.drop(columns=["slide_id"], inplace=True)

    for mag in metadata["magnification"].unique():
        filtered_metadata = metadata[metadata["magnification"] == mag]
        with h5py.File(
            f"{args.feat_folder}/{slide_id}/{mag}x_features.h5", "a"
        ) as h5_file:
            if "metadata" in h5_file.keys():
                del h5_file["metadata"]
            h5_file.create_dataset(
                "metadata",
                data=filtered_metadata.to_records(index=False),
                compression="gzip",
            )

def collate_features(batch):
    # img = torch.cat([torch.from_numpy(item[0]) for item in batch], dim = 0)
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

def store_features(args, features_dict: Dict[str, torch.Tensor], slide_id: str):
    """Store extracted features into HDF5 files categorized by model and magnification."""
    slide_dir = os.path.join(args.feat_folder, slide_id)
    os.makedirs(slide_dir, exist_ok=True)

    for model_type, mag_features in features_dict.items():
        for mag, features in mag_features.items():
            h5_file_path = os.path.join(slide_dir, f"{mag}x_features.h5")
            with h5py.File(h5_file_path, "a") as h5_file:  # Open in append mode
                model_name = str(model_type).upper()
                features_dataset_name = f"{model_name}_features"
                indices_dataset_name = f"{model_name}_indices"

                features_array = torch.stack(list(features.values())).numpy()
                indices_array = np.array(list(features.keys()), dtype="int")

                # Check if the dataset already exists
                if features_dataset_name in h5_file:
                    # If exists, replace the dataset
                    del h5_file[features_dataset_name]
                if indices_dataset_name in h5_file:
                    del h5_file[indices_dataset_name]

                # Create datasets for features and indices
                h5_file.create_dataset(
                    features_dataset_name,
                    data=features_array,
                    dtype="float32",
                    compression="gzip",
                )
                h5_file.create_dataset(
                    indices_dataset_name,
                    data=indices_array,
                    dtype="int",
                    compression="gzip",
                )

def process_tiles_pt(args, slide_id: str, models: Dict[str, nn.Module], progress=0):
    '''
    Input:
        args: the arguments
        slide_id: str, the slide id
        models: Dict[str, nn.Module]
        progress: float, the progress of the processing
    '''
    # model_name = list(models.keys())[0]

    ## Added by SY: search for WSI files based on slide_id
    
    if args.csv_path:
        wsi_df = pd.read_csv(args.csv_path)
        all_files = wsi_df.loc[wsi_df[args.id_col] == slide_id, args.wsi_col].values
    elif args.img_format == "wsi":
        ndpi_files_subdirs = glob.glob(f"{args.wsi_folder}/**/*{slide_id}.ndpi", recursive=True)
        svs_files_subdirs = glob.glob(f"{args.wsi_folder}/**/*{slide_id}.svs", recursive=True)
        mrxs_files_subdirs = glob.glob(f"{args.wsi_folder}/**/*{slide_id}.mrxs", recursive=True)
        all_files = ndpi_files_subdirs + svs_files_subdirs + mrxs_files_subdirs
    else:
        tif_files_subdirs = glob.glob(f"{args.wsi_folder}/**/*{slide_id}.tif", recursive=True)
        tiff_files_subdirs = glob.glob(f"{args.wsi_folder}/**/*{slide_id}.tiff", recursive=True)
        all_files = tif_files_subdirs + tiff_files_subdirs
    
    if len(all_files) == 0:
        raise ValueError(f"No WSI files found in {args.wsi_folder}")
    if len(all_files) > 1:
        # raise ValueError(f"Multiple WSI files found in {args.wsi_folder}")
        Warning(f"Multiple WSI files found in {args.wsi_folder}, using the first one")
    wsi_path = all_files[0]
    ##module

    h5_path = f"{args.patch_folder}/coords/{slide_id}.h5"
    # wsi_path = f"{args.wsi_folder}/{slide_id}.svs"
    if args.stain_norm:
        stain_norm_str = '(stain_norm)'
    else:
        stain_norm_str = ''
        
    feat_h5_dirs = {model_name: f"{args.feat_folder}/{model_name}/{args.target_mag}X/h5_files{stain_norm_str}" for model_name in models.keys()}
    feat_pt_dirs = {model_name: f"{args.feat_folder}/{model_name}/{args.target_mag}X/pt_files{stain_norm_str}" for model_name in models.keys()}
    [os.makedirs(feat_h5_dir, exist_ok=True) for feat_h5_dir in feat_h5_dirs.values()]
    [os.makedirs(feat_pt_dir, exist_ok=True) for feat_pt_dir in feat_pt_dirs.values()]
    
    wsi = openslide.OpenSlide(wsi_path)
    transforms = get_transforms()
    
    if args.stain_norm:
        ## find the thumbnail mask for stain normalization
        thumbnail_mask = os.path.join(args.patch_folder,'masks',f'{slide_id}.png')
        assert os.path.exists(thumbnail_mask), f"Thumbnail mask not found for {slide_id}"
    else:
        thumbnail_mask = None
    wsi_dataset = WSIDataset(
        args, h5_path, wsi,
        custom_transforms=transforms,thumbnail_mask_path=thumbnail_mask) 
    ####
    ## If max_num_patches is set and the dataset length is greater than max_num_patches, select a random subset
    if args.max_num_patches is not None and len(wsi_dataset) > args.max_num_patches:
        print(f"{slide_id} has {len(wsi_dataset)} patches, which is greater than the maximum number of patches to process ({args.max_num_patches})")
        print(f"Selecting a random subset of {args.max_num_patches} patches")
        indices = np.random.choice(len(wsi_dataset), args.max_num_patches, replace=False)
        wsi_dataset = Subset(wsi_dataset, indices)
    

    kwargs = {'num_workers': args.n_workers, 'pin_memory': True} if args.device == "cuda" else {}
    loader = DataLoader(dataset=wsi_dataset,
                        batch_size=args.batch_size,
                        collate_fn=collate_features, **kwargs)

    print(f"Processing {slide_id}.h5")
    modes = {model_name: 'w' for model_name in models.keys()}
    pbar = tqdm(loader,mininterval=TQDM_MIN_INTERVAL)
    for count, (batch, coords) in enumerate(pbar):
        with torch.no_grad():
            batch = batch.to(args.device, non_blocking=True)
            out = get_features(batch, models)
            for feat_name in out.keys():
                features = out[feat_name]
                features = features.cpu().numpy()
                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(f"{feat_h5_dirs[feat_name]}/{slide_id}.h5", asset_dict, attr_dict={'feat_name': feat_name}, mode=modes[feat_name])
                modes[feat_name] = 'a'
                pbar.set_description(f'Progress {progress*100:.1f}%, Batch {count}/{len(loader)}, {count * args.batch_size} files processed', refresh=False)
    ## convert h5 files to pt files
    for model_name in models.keys():
        feat_h5_dir = feat_h5_dirs[model_name]
        feat_pt_dir = feat_pt_dirs[model_name]
        with h5py.File(f"{feat_h5_dir}/{slide_id}.h5", "r") as h5_file:
            features = h5_file['features'][:]
        features = torch.from_numpy(features)
        print('features size: ', features.shape)
        print(f"Saving features to {feat_pt_dir}/{slide_id}.pt")

        torch.save(features, f"{feat_pt_dir}/{slide_id}.pt")

def main():
    args = parse_args()
    setup_folders(args)
    initialize_db(args)

    print("=" * 50)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(vars(args))
    print("=" * 50)

    all_h5s = load_all_h5s(args)
    print(f"Found {len(all_h5s)} available slide bags to process.")

    print(f"Split items into {args.n_parts} parts, processing part {args.part}")
    total_csvs = len(all_h5s)
    part_size = math.ceil(total_csvs / args.n_parts)
    start_index = args.part * part_size
    end_index = min(start_index + part_size, total_csvs)
    all_h5s = all_h5s[start_index:end_index]
    slide_ids = [os.path.basename(csv)[:-3] for csv in all_h5s]
    print(f"Process slide indices [{start_index}:{end_index}]")

    models = load_models(args)
    
    success_data = load_success_data(args)
    requested_experts = set(str(m).upper() for m in args.models)
    

    for idx, slide_id in enumerate(slide_ids):
        # skip if already processed
        if slide_id in success_data["slide_id"].values:
            processed_experts = set(
                success_data[success_data["slide_id"] == slide_id]["experts"]
                .str.split(",")
                .values[0]
            )
            if requested_experts.issubset(processed_experts):
                print(
                    f"Skipping already processed slide_id with all requested experts: {slide_id}"
                )
                continue
            else:
                print(f"Processing missing experts for slide_id: {slide_id}")
        try:
            process_tiles_pt(args, slide_id, models, progress=idx/len(slide_ids))
        except Exception as e:
            print(f"Error processing {slide_id}: {e}")
            tb = traceback.extract_tb(e.__traceback__)
            filename, line, func, text = tb[-1]  # Get last entry in traceback for exact error location
            print(f"An error occurred in file '{filename}' on line {line} in '{func}'")
            print(f"Error message:{text}")
            continue

        update_success(args, slide_id, [m.upper() for m in models.keys()])


if __name__ == "__main__":
    main()

# python create_features_bao.py \
#     --patch_folder /n/scratch/users/b/bal753/DFCI_MOTSU \
#     --patch_size 224 \
#     --stride 224 \
#     --output_size 224 \
#     --tissue_threshold 5 \
#     --magnifications 20\
#     --n_workers 16 --only_coords