import argparse
import datetime
import glob
import logging
import math
import os
import pprint
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import PIL
import PIL.Image
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm
from skimage.filters import threshold_multiotsu

MAX_VISUALIZE_PATCHES = 2000
Slide = openslide.OpenSlide
# Modified by bao
# 1. add support of reading csv files directly;
# 2. add metadata to the coord files;
# 3. change the folder organization (coords, masks, foreground, thumbnail);
# 4. change the foreground extraction method;

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose magnification level and patch parameters."
    )

    parser.add_argument(
        "--slide_folder",
        type=str,
        default="/n/data2/hms/dbmi/kyu/lab/bal753/WSI_for_debug",
        help="Root slides folder.",
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help="""Path to the csv file containing the slide paths.
        If not provided, the script will search for all WSI files in the slide_folder.""",
    )
    parser.add_argument(
        '--wsi_col',
        type=str,
        default='WSI_path',
        help="""Column name in the csv file containing the slide paths.""",
    )
    parser.add_argument(
        '--mag_col',
        type=str,
        default='mag',
        help="""Column name in the csv file containing the magnification level of the slides.""",
    )
    
    parser.add_argument(
        "--patch_folder",
        type=str,
        default="/n/data2/hms/dbmi/kyu/lab/shl968/tile_dataset_for_debug",
        help="Root patch folder.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=224,
        help="The stride in x and y (default 224)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="Patch size (default 224)",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=224,
        help="Output size of each patch (default 224)",
    )
    parser.add_argument(
        "--tissue_threshold",
        type=int,
        default=80,
        help="Minimum tissue percentage threshold to save patches (default 80)",
    )
    parser.add_argument(
        "--magnifications",
        type=int,
        nargs="+",
        default=[40, 20, 10],
        help="Magnifications to extract patches for (default 40 20 10)",
    )
    parser.add_argument(
        "--keep_top_n",
        type=int,
        default=None,
        help="Keep only the top N patches with the highest tissue percentage (default None, which keeps all)",
    )
    parser.add_argument(
        "--keep_random_n",
        type=int,
        default=None,
        help="Keep only a maximum of random N patches (default None, which keeps all)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers to use for processing patches in parallel(default 1)",
    )
    parser.add_argument(
        "--n_parts",
        type=int,
        default=1,
        help="The number of parts to split the slides into (default 1)",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=0,
        help="The part of the slides to process (default 0)",
    )
    parser.add_argument(
        "--only_coords",
        action="store_true",
        help="Only extract coordinates to a <slide_id>.h5 file.",
    )
    parser.add_argument(
        "--use_center_mask",
        action="store_true",
        help="Instead of using otsu thresholding, use a center mask to define the viable patch area.",
    )
    parser.add_argument(
        "--center_mask_height",
        type=float,
        default=0.5,
        help="The height of the center mask as a fraction of the image height (default 0.5).",
    )
    ## Added options for reading tiff files
    parser.add_argument(
        "--img_format",
        type=str,
        default="wsi",
        choices=["wsi", "tiff"],
        help="The format of the input images (default wsi). If tiff, the script will use the magnification from the argument.")
    parser.add_argument(
        "--tiff_mag",
        type=int,
        default=40,
        help="The magnification of the tiff images (default 40, for Phillips Scanners).")
    parser.add_argument(
        "--wsi_mag",
        type=int,
        default=-1,
        help="The magnification of the wsi images (default -1).")
    ##
    return parser.parse_args()


@dataclass
class PatchConfig:
    slide_folder: str
    patch_folder: str
    patch_size: int = 1000
    stride: int = 250
    output_size: int = 224
    tissue_threshold: int = 80
    magnifications: List[int] = field(default_factory=lambda: [40, 20, 10])
    keep_top_n: Optional[int] = None
    keep_random_n: Optional[int] = None
    n_workers: int = 1
    n_parts: int = 1
    part: int = 0
    only_coords: bool = False
    use_center_mask: bool = False
    center_mask_height: float = 0.5


@dataclass
class PatchPack:
    magnifications: List[int]
    images: List[PIL.Image.Image]
    tissue_percentage: float
    center_x: int
    center_y: int
    patch_size: int
    stride: int
    output_size: int


class H5Saver:
    """
    H5Saver: Efficient HDF5-based storage for large-scale image patch processing.

    This class provides thread-safe, low-memory operations for saving and managing
    image patches and associated metadata in HDF5 files. It's designed for
    scenarios involving large datasets or memory-constrained environments.

    Key Features:
    - Very low memory foot print (stores one patch pack at a time)
    - Append-only writes for efficient storage of image patches
    - Concurrent access support via threading.Lock
    - Dynamic dataset creation and extension
    - Metadata storage alongside image data
    - Filtering capability to retain top N patches based on tissue percentage

    Example:
        >>> saver = H5Saver('output.h5')
        >>> saver.push_pack(patch_pack)
        >>> # after processing all patches, keep only top 1000 for each magnification
        >>> saver.keep_top_n(1000, magnifications=[20, 40, 60])
        >>> # get the total number of patches
        >>> total_patches = len(saver)

    Note:
    The 'keep_top_n' method modifies the file in-place and cannot be undone.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.file = h5py.File(self.file_path, "a")
        self.lock = Lock()

    def push_pack(self, pack: PatchPack):
        # aquire mutex
        with self.lock:
            # append the images to the corresponding datasets
            img: PIL.Image.Image
            mag: int
            for img, mag in zip(pack.images, pack.magnifications):
                dataset_name = str(mag)
                img_array = np.asarray(img)
                if dataset_name not in self.file:
                    self.file.create_dataset(
                        dataset_name,
                        data=img_array[np.newaxis, ...],
                        maxshape=(None, *img_array.shape),
                        chunks=True,
                        compression="gzip",
                        compression_opts=9,
                    )
                else:
                    dataset = self.file[dataset_name]
                    current_size = dataset.shape[0]
                    dataset.resize(current_size + 1, axis=0)
                    dataset[current_size] = img_array

            # write meta data
            if "metadata" not in self.file:
                dtype = np.dtype(
                    [
                        ("tissue_percentage", float),
                        ("center_x", int),
                        ("center_y", int),
                        ("patch_size", int),
                        ("stride", int),
                        ("output_size", int),
                    ]
                )
                self.file.create_dataset(
                    "metadata",
                    (0,),
                    dtype=dtype,
                    maxshape=(None,),
                    chunks=True,
                    compression="gzip",
                    compression_opts=9,
                )

            metadata = self.file["metadata"]
            current_size = metadata.shape[0]
            metadata.resize(current_size + 1, axis=0)
            metadata[current_size] = (
                pack.tissue_percentage,
                pack.center_x,
                pack.center_y,
                pack.patch_size,
                pack.stride,
                pack.output_size,
            )

    def keep_top_n(self, n: int, magnifications: List[int]):
        """
        Filter and keep only the top N patches for each specified magnification.

        This method modifies the HDF5 file in-place, deleting patches with lower
        tissue percentages. This operation cannot be undone.

        Args:
            n (int): Number of top patches to keep for each magnification.
            magnifications (List[int]): List of magnification levels to process.
        """
        with self.lock:
            metadata = self.file["metadata"]

            for mag in magnifications:
                dataset_name = str(mag)
                if dataset_name not in self.file:
                    continue

                dataset = self.file[dataset_name]

                tissue_percentages = metadata["tissue_percentage"][: len(dataset)]

                # get indices of top n tissue percentages
                top_n_indices = np.argsort(tissue_percentages)[-n:]
                top_n_indices.sort()

                # delete images
                temp_dataset = self.file.create_dataset(
                    f"{dataset_name}_temp",
                    data=dataset[top_n_indices],
                    maxshape=(None, *dataset.shape[1:]),
                    chunks=True,
                    compression="gzip",
                    compression_opts=9,
                )
                del self.file[dataset_name]
                self.file[dataset_name] = temp_dataset
                del self.file[f"{dataset_name}_temp"]

            # update the metadata as well
            top_n_indices_all = np.argsort(metadata["tissue_percentage"])[-n:]
            top_n_indices_all.sort()
            new_metadata = metadata[top_n_indices_all]

            del self.file["metadata"]
            self.file.create_dataset(
                "metadata",
                data=new_metadata,
                maxshape=(None,),
                chunks=True,
                compression="gzip",
                compression_opts=9,
            )

            self.file.flush()

    def keep_random_n(self, n: int):
        """
        Randomly select and keep only N patches across all magnifications.

        This method modifies the HDF5 file in-place, randomly deleting patches
        to reduce the total count to N. This operation cannot be undone.
        """
        with self.lock:
            metadata = self.file["metadata"]
            total_patches = len(metadata)

            if n >= total_patches:
                return

            # randomly select indices to keep
            indices_to_keep = np.random.choice(total_patches, n, replace=False)
            indices_to_keep.sort()

            for dataset_name in self.file.keys():
                if dataset_name == "metadata":
                    continue

                dataset = self.file[dataset_name]
                new_data = dataset[indices_to_keep]

                del self.file[dataset_name]
                self.file.create_dataset(
                    dataset_name,
                    data=new_data,
                    chunks=True,
                    compression="gzip",
                    compression_opts=9,
                )

            new_metadata = metadata[indices_to_keep]
            del self.file["metadata"]
            self.file.create_dataset(
                "metadata",
                data=new_metadata,
                chunks=True,
                compression="gzip",
                compression_opts=9,
            )
            self.file.flush()

    def __len__(self):
        with self.lock:
            if self.file is None:
                return 0
            if "metadata" not in self.file:
                return 0
            return len(self.file["metadata"])


def setup_folders(args: PatchConfig):
    os.makedirs(args.patch_folder, exist_ok=True)
    os.makedirs(f"{args.patch_folder}/coords", exist_ok=True)

def get_mag(wsi):
    mag_ref = [20, 40]
    mag_li = []
    if 'tiff.XResolution' in wsi.properties.keys():
        x_res = float(wsi.properties['tiff.XResolution']) / 10000
        y_res = float(wsi.properties['tiff.YResolution']) / 10000
        assert x_res == y_res
        mag = 10*x_res
        mag = min(mag_ref, key=lambda x: abs(x - mag))
        mag_li.append(mag)
    if 'openslide.mpp-x' in wsi.properties.keys():    # represent the microns per pixel in the X and Y dimensions
        x_spacing = float(wsi.properties['openslide.mpp-x'])
        y_spacing = float(wsi.properties['openslide.mpp-y'])
        assert x_spacing == y_spacing
        mag = 10/x_spacing
        mag = min(mag_ref, key=lambda x: abs(x - mag))
        mag_li.append(mag)
    if 'aperio.MPP' in wsi.properties.keys():    # represents microns per pixel but is typically a single value
        mpp = float(wsi.properties['aperio.MPP'])
        mag = 10/mpp
        mag = min(mag_ref, key=lambda x: abs(x - mag))
        mag_li.append(mag)
    if 'aperio.AppMag' in wsi.properties.keys():  # openslide doesn't set objective-power for all SVS files: https://github.com/openslide/openslide/issues/247
        mag = float(wsi.properties.get("aperio.AppMag", None))
        mag_li.append(int(mag))
    if 'openslide.objective-power' in wsi.properties.keys():  # openslide doesn't set objective-power for all SVS files: https://github.com/openslide/openslide/issues/247
        mag = float(wsi.properties.get("openslide.objective-power", None))
        mag_li.append(int(mag))
        
    if len(mag_li) == 0:
        raise ValueError(f"Magnification not founds")
    elif len(mag_li) != 0 and len(np.unique(mag_li)) == 1:
        return mag_li[0]
    else:
        raise ValueError(f"Magnification conflict: {np.unique(mag_li)}")

def store_available_coords(args: PatchConfig, slide_id, coords: np.ndarray,
                           mag=None):
    if type(coords) != np.ndarray:
        coords = np.array(coords, dtype=[("x", np.int32), ("y", np.int32)])

    with h5py.File(f"{args.patch_folder}/coords/{slide_id}.h5", "w") as f:
        _ = f.create_dataset("coords", data=coords)
        # write meta data
        if "metadata" not in f:
            dtype = np.dtype(
                [
                    ("magnification", int),
                    ("patch_size", int),
                    ("stride", int),
                    ("output_size", int),
                ]
            )
            f.create_dataset(
                "metadata",
                (0,),
                dtype=dtype,
                maxshape=(None,),
                chunks=True,
                compression="gzip",
                compression_opts=9,
            )

        metadata = f["metadata"]
        current_size = metadata.shape[0]
        metadata.resize(current_size + 1, axis=0)
        metadata[current_size] = (
            mag,
            args.patch_size,
            args.stride,
            args.output_size,
        )


def visualize_patches(args: PatchConfig, slide_path: str, target_mag: int = 20):
    slide_id = get_slide_id(slide_path)
    h5_file = h5py.File(f"{args.patch_folder}/coords/{slide_id}.h5", "r")
    metadata = h5_file["metadata"]
    
    wsi = openslide.OpenSlide(slide_path)
    thumbnail = get_thumbnail(wsi)
    level_0_dim = wsi.dimensions

    level_0_mag = int(wsi.properties.get("openslide.objective-power", 40))
    scaling_factor = level_0_mag / target_mag
    adjusted_patch_size = int(args.patch_size * scaling_factor)
    downsample = level_0_dim[0] / thumbnail.shape[1]
    thumbnail_patch_size = int(adjusted_patch_size / downsample)

    # Adjust the patch coordinates for the thumbnail scaling
    coords = list(zip(metadata["center_x"][:], metadata["center_y"][:]))

    print(f"Overlay {len(coords)} patches at magnification {target_mag}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(thumbnail)
    linewidth = 2 if len(coords) < 100 else 0.5

    if len(coords) > 2000:
        print(
            "Warning: Got a large number of patches for visualization. This might take a while."
        )
    if len(coords) > MAX_VISUALIZE_PATCHES:
        print(
            f"Warning: Got more than {MAX_VISUALIZE_PATCHES} patches, will only visualize a subset."
        )
        indices = np.random.choice(len(coords), MAX_VISUALIZE_PATCHES, replace=False)
        coords = [coords[i] for i in indices]

    for x, y in coords:
        # Adjust coordinates for thumbnail display
        top_left_x = int((x - adjusted_patch_size / 2) / downsample)
        top_left_y = int((y - adjusted_patch_size / 2) / downsample)
        rect = Rectangle(
            (top_left_x, top_left_y),
            thumbnail_patch_size,
            thumbnail_patch_size,
            linewidth=linewidth,
            edgecolor="green",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.axis("off")
    # output_path = f"{args.patch_folder}/{slide_id}.png"
    output_path = f"{slide_id}.png"
    plt.savefig(output_path)
    plt.close()

    h5_file.close()


def setup_logging(args):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    today_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"{args.patch_folder}/{today_date}.log"
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    # redirect outputs to file logger
    class StreamToLogger:
        def __init__(self, logger, log_level):
            self.logger = logger
            self.log_level = log_level
            self.linebuf = ""

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())

        def flush(self):
            pass

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)


def load_slides(args) -> List[str]:

    if args.csv_path is None:
        if args.img_format == "wsi":
            ndpi_files_direct = glob.glob(f"{args.slide_folder}/*.ndpi")
            svs_files_direct = glob.glob(f"{args.slide_folder}/*.svs")
            mrxs_files_direct = glob.glob(f"{args.slide_folder}/*.mrxs")
            ndpi_files_subdirs = glob.glob(f"{args.slide_folder}/**/*.ndpi", recursive=True)
            svs_files_subdirs = glob.glob(f"{args.slide_folder}/**/*.svs", recursive=True)
            mrxs_files_subdirs = glob.glob(f"{args.slide_folder}/**/*.mrxs", recursive=True)
            all_files = (
                ndpi_files_direct
                + svs_files_direct
                + ndpi_files_subdirs
                + svs_files_subdirs
                + mrxs_files_direct
                + mrxs_files_subdirs
            )
            df = pd.DataFrame({args.wsi_col: all_files})
        else:
            tiff_files_direct = glob.glob(f"{args.slide_folder}/*.tiff")
            tiff_files_subdirs = glob.glob(f"{args.slide_folder}/**/*.tiff", recursive=True)
            all_files = tiff_files_direct + tiff_files_subdirs
            df = pd.DataFrame({args.wsi_col: all_files})
    else:
        df = pd.read_csv(args.csv_path)
        all_files = list(df[args.wsi_col].values)

    all_files = list(set(all_files))
    all_files = sorted([f for f in all_files if os.path.isfile(f)])
    return all_files, df


def load_success_ids(args) -> Set[str]:
    success_txt = f"{args.patch_folder}/success.txt"
    success_ids = set()
    if os.path.exists(success_txt):
        with open(success_txt) as f:
            success_ids = {line.strip() for line in f}
    return success_ids


def clean_unfinished(args: PatchConfig, slide_id: str):
    if os.path.exists(f"{args.patch_folder}/coords/{slide_id}"):
        shutil.rmtree(f"{args.patch_folder}/coords/{slide_id}")
    if os.path.exists(f"{args.patch_folder}/coords/{slide_id}.h5"):
        print("Clean up existing h5 file")
        os.remove(f"{args.patch_folder}/coords/{slide_id}.h5")
    if os.path.exists(f"{args.patch_folder}/coords/{slide_id}.csv"):
        os.remove(f"{args.patch_folder}/coords/{slide_id}.csv")
    if os.path.exists(f"{args.patch_folder}/coords/{slide_id}.zip"):
        os.remove(f"{args.patch_folder}/coords/{slide_id}.zip")


def get_slide_id(slide_path: str) -> str:
    fname = os.path.basename(slide_path)
    # tcga files have 10 "-" separated parts
    # is_tcga = len(fname.split("-")) == 10
    # if is_tcga:    ## disabled for compatibility with other datasets
    #     return os.path.basename(slide_path).split(".")[1]
    # Check if the filename contains multiple dots in it, ex cytology slide contains date : 0081__-__02.14.20.ndpi
    fname, _ = os.path.splitext(fname)
    return fname


def calculate_tissue_percentage(
    patch: PIL.Image.Image,
    lower_hsv: np.ndarray = np.array([0.5 * 255, 0.2 * 255, 0.2 * 255]),
    upper_hsv: np.ndarray = np.array([1.0 * 255, 0.7 * 255, 1.0 * 255]),
    patch_size: int = 224,
) -> float:
    hsv_image = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2HSV)
    tissue_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    tissue_area = np.count_nonzero(tissue_mask)
    total_area = patch_size**2
    return (tissue_area / total_area) * 100


def get_thumbnail(wsi: Slide, downsample: int = 16,max_width=3000) -> np.ndarray:
    full_size = wsi.dimensions
    thumbnail_size = [int(full_size[0] / downsample), int(full_size[1] / downsample)]
    
    while any([x>max_width for x in thumbnail_size]):
        downsample = downsample * 2
        thumbnail_size = [int(full_size[0] / downsample), int(full_size[1] / downsample)]

    img_rgb = np.array(
        wsi.get_thumbnail(tuple(thumbnail_size))
    )
    return img_rgb


def get_tissue_mask(args: PatchConfig, img_rgb: np.ndarray):
    # r_channel, g_channel, b_channel = cv2.split(img_rgb)

    # r_channel = np.clip(r_channel * 1.5, 0, 255).astype(np.uint8)
    # g_channel = np.clip(g_channel * 1.1, 0, 255).astype(np.uint8)
    # b_channel = np.clip(b_channel * 1.1, 0, 255).astype(np.uint8)

    # img_enhanced = cv2.merge([r_channel, g_channel, b_channel])
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Define H channel range to enhance red and pink
    lower_red = np.array([160, 50, 50])   # Lower bound for red
    upper_red = np.array([180, 255, 255])  # Upper bound for red
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)  # pixel wih colour red is assigned with 0

    # Define H channel range to suppress blue and green
    lower_blue_green = np.array([0, 50, 50])   # Lower bound for blue and green
    upper_blue_green = np.array([120, 255, 255])  # Upper bound for blue and green
    mask_blue_green = cv2.inRange(hsv_image, lower_blue_green, upper_blue_green)

    # Create the final mask to enhance red and pink, and suppress blue and green
    final_mask = cv2.bitwise_or(mask_red, cv2.bitwise_not(mask_blue_green))

    # Apply the mask
    img_enhanced = cv2.bitwise_and(img_rgb, img_rgb, mask=final_mask)

    hsv_image = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2HSV)
    img_med = cv2.medianBlur(hsv_image[:, :, 1], 11)
    thresholds = threshold_multiotsu(img_med, classes=3)
    # _, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thres = max(thresholds[0], 15)
    tissue_mask = (img_med>thres).astype(np.uint8)
    tissue_mask[tissue_mask!=0] = 255
    k = np.ones((7, 7), dtype=np.uint8)
    tissue_mask = cv2.morphologyEx(src=tissue_mask, kernel=k, op=cv2.MORPH_OPEN, iterations=3)
    tissue_mask = cv2.morphologyEx(src=tissue_mask, kernel=k, op=cv2.MORPH_CLOSE, iterations=3)

    if args.use_center_mask:
        # Create a mask with a central rectangle activated
        center_mask = np.zeros_like(tissue_mask)
        height, width = center_mask.shape
        rect_height = int(height * args.center_mask_height)
        rect_width = rect_height
        start_x = (width - rect_width) // 2
        start_y = (height - rect_height) // 2
        center_mask[start_y : start_y + rect_height, start_x : start_x + rect_width] = 1
        tissue_mask = cv2.bitwise_and(tissue_mask, tissue_mask, mask=center_mask)

    return tissue_mask


def find_tissue_patches(args: PatchConfig, wsi: Slide, slide_id=None, mag=None) -> List[Tuple[int, int]]:
    target_mag = max(args.magnifications)
    patch_size = args.patch_size
    stride = args.stride
    tissue_threshold = args.tissue_threshold / 100
    thumbnail = get_thumbnail(wsi)
    if os.path.exists(f'{args.patch_folder}/masks/{slide_id}.png'):
        tissue_mask = np.array(Image.open(f'{args.patch_folder}/masks/{slide_id}.png').convert('L'))
        tissue_mask[tissue_mask>0] = 255
        t_h, t_w = tissue_mask.shape
        if t_h != thumbnail.shape[0] or t_w != thumbnail.shape[1]:
            tissue_mask = cv2.resize(tissue_mask, (thumbnail.shape[1], thumbnail.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        tissue_mask = get_tissue_mask(args, thumbnail)

    level_0_dim = wsi.dimensions
    if mag is None:
        if args.img_format == "tiff":
            level_0_mag = args.tiff_mag
        else:
            level_0_mag = get_mag(wsi)
    else:
        level_0_mag = mag

    os.makedirs(f'{args.patch_folder}/thumbnail', exist_ok=True)
    os.makedirs(f'{args.patch_folder}/masks', exist_ok=True)
    os.makedirs(f'{args.patch_folder}/foreground', exist_ok=True)

    Image.fromarray(thumbnail).save(f'{args.patch_folder}/thumbnail/{slide_id}.png')
    thumbnail_mask_rgb = Image.fromarray(thumbnail*(tissue_mask[..., np.newaxis]!=0))
    thumbnail_mask_rgb.save(f'{args.patch_folder}/masks/{slide_id}.png')
    # Find contours
    contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw contours on
    cv2.drawContours(thumbnail, contours, -1, (255, 0, 0), thickness=6)  # white contours
    Image.fromarray(thumbnail).save(f'{args.patch_folder}/foreground/{slide_id}.png')

    # compute the downsample from the tissue mask dimensions to level_0_dim
    downsample = level_0_dim[0] / tissue_mask.shape[1]

    # scaling factor (level_0_mag to target_mag)
    scaling_factor = level_0_mag / target_mag

    # Adjust the patch size and stride based on the magnification difference
    adjusted_patch_size = int(patch_size * scaling_factor)
    adjusted_stride = int(stride * scaling_factor)
    valid_patches = []
    # Iterate over the full image at level 0 with steps of adjusted_stride
    for y in range(0, level_0_dim[1] - adjusted_patch_size + 1, adjusted_stride):
        for x in range(0, level_0_dim[0] - adjusted_patch_size + 1, adjusted_stride):
            # Map the coordinates to the tissue mask scale
            mask_x = int(x / downsample)
            mask_y = int(y / downsample)
            mask_patch_size = int(adjusted_patch_size / downsample)

            # Ensure we don't go out of bounds in the tissue mask
            if (
                mask_x + mask_patch_size > tissue_mask.shape[1]
                or mask_y + mask_patch_size > tissue_mask.shape[0]
            ):
                continue

            # Check if average intensity is above the threshold
            mask_area = tissue_mask[
                mask_y : mask_y + mask_patch_size, mask_x : mask_x + mask_patch_size
            ]
            if np.mean(mask_area) > 255 * tissue_threshold:
                valid_patches.append((x, y))
    return valid_patches


def extract_patches(
    slide_path: str,
    center_location: Tuple[int, int],
    patch_size: int,
    output_size: int,
    target_mags: List[int] = [40, 20, 10],
) -> Tuple[Dict[int, PIL.Image.Image], Dict[int, float]]:
    if type(patch_size) not in [tuple, list]:
        patch_size = (patch_size, patch_size)
    if type(output_size) not in [tuple, list]:
        output_size = (output_size, output_size)

    slide = openslide.OpenSlide(slide_path)
    highest_mag = float(slide.properties["openslide.objective-power"])
    native_magnifications = {
        highest_mag / slide.level_downsamples[level]: level
        for level in range(slide.level_count)
    }
    patches = {}
    percentages = {}
    for target_mag in target_mags:
        if target_mag in native_magnifications:
            level = native_magnifications[target_mag]
            downsample = slide.level_downsamples[level]
            # calculate the center location at the native resolution
            center_x = int(center_location[0] * slide.level_downsamples[0])
            center_y = int(center_location[1] * slide.level_downsamples[0])
            # fix the center location to the top-left corner of the patch
            location = (
                int(center_x - patch_size[0] // 2 * downsample),
                int(center_y - patch_size[1] // 2 * downsample),
            )
            patch = slide.read_region(location, level, patch_size)
        else:
            nearest_higher_mag = max(
                [mag for mag in native_magnifications if mag > target_mag],
                default=highest_mag,
            )
            nearest_higher_level = native_magnifications[nearest_higher_mag]
            scale_factor = nearest_higher_mag / target_mag
            extract_size = (
                round(patch_size[0] * scale_factor),
                round(patch_size[1] * scale_factor),
            )
            # calculate the center location at the highest resolution
            center_x = int(center_location[0] * slide.level_downsamples[0])
            center_y = int(center_location[1] * slide.level_downsamples[0])
            new_location = (
                round(center_x - extract_size[0] / 2),
                round(center_y - extract_size[1] / 2),
            )
            patch = slide.read_region(new_location, nearest_higher_level, extract_size)

        patch = patch.resize(output_size, Image.BILINEAR)
        patches[target_mag] = patch.convert("RGB")
        percentages[target_mag] = calculate_tissue_percentage(patches[target_mag])
    return patches, percentages


def process_patch(slide_path: str, x: int, y: int, args) -> Optional[PatchPack]:
    """
    Processes a patch and returns a tuple of the patch, its coordinates, and the tissue percentage or None if the patch should be discarded.
    """
    try:
        patches: dict
        patches, tissue_percentages = extract_patches(
            slide_path, (x, y), args.patch_size, args.output_size, args.magnifications
        )
        if (
            calculate_tissue_percentage(patches[max(args.magnifications)])
            < args.tissue_threshold
        ):
            return None

        images = []
        magnifications = []
        for mag in patches:
            images.append(patches[mag])
            magnifications.append(mag)
        pack = PatchPack(
            magnifications=magnifications,
            images=images,
            center_x=x,
            center_y=y,
            tissue_percentage=tissue_percentages[max(args.magnifications)],
            patch_size=args.patch_size,
            stride=args.stride,
            output_size=args.output_size,
        )
        return pack
    except Exception as e:
        print(f"Failed to process patch ({x},{y}) at {slide_path}: {e}")
    return None


def process_slide(args: PatchConfig, slide_path: str, patch_coords: list):
    slide_id = get_slide_id(slide_path)
    total_patches = len(patch_coords)

    h5_path = f"{args.patch_folder}/coords/{slide_id}.h5"
    h5_saver = H5Saver(h5_path)
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor, tqdm(
        total=total_patches, desc=f"Processing {slide_id}"
    ) as pbar:
        tasks = {
            executor.submit(process_patch, slide_path, x, y, args): (
                x,
                y,
            )
            for x, y in patch_coords
        }
        for future in as_completed(tasks):
            result = future.result()
            if result:
                pack: PatchPack = result
                h5_saver.push_pack(pack)
            pbar.update(1)

            # check every now and then for keep_top_n

    if args.keep_top_n:
        h5_saver.keep_top_n(args.keep_top_n, args.magnifications)
    if args.keep_random_n:
        h5_saver.keep_random_n(args.keep_random_n)
    print(f"Saved {len(h5_saver)} patches for slide id {slide_id}")

def main():
    args = parse_args()
    setup_folders(args)
    # setup_logging(args)

    print("=" * 50)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(vars(args))
    print("=" * 50)

    # === Load all WSI files ===
    all_slides, all_df  = load_slides(args)

    print(f"Loaded {len(all_slides)} slides.")

    success_ids = load_success_ids(args)
    print(f"Split slides into {args.n_parts} parts, processing part {args.part}")
    total_slides = len(all_slides)
    part_size = math.ceil(total_slides / args.n_parts)
    start_index = args.part * part_size
    end_index = min(start_index + part_size, total_slides)
    all_slides = all_slides[start_index:end_index]
    print(f"Process slide indices [{start_index}:{end_index}]")
    all_slides = [
        slide for slide in all_slides if get_slide_id(slide) not in success_ids
    ]
    print("Filtered out already previously processed slides.")

    ##
    for slide_path in all_slides:
        slide_id = get_slide_id(slide_path)

        if slide_id in success_ids or slide_id.startswith("$"): # some svs files in RoswellParkCancerCenter start with $
            continue
        clean_unfinished(args, slide_id)

        try:
        # if True:
            print(f"Start processing slide: {slide_id}")
            wsi = openslide.OpenSlide(slide_path)
            if args.mag_col in all_df.columns:
                mag = all_df.loc[all_df[args.wsi_col]==slide_path, args.mag_col].values[0]
            elif args.img_format == 'wsi' and args.wsi_mag != -1:
                mag = args.wsi_mag
            elif args.img_format == "tiff":
                mag = args.tiff_mag
            else:
                mag = get_mag(wsi)
            patch_coords = find_tissue_patches(args, wsi, slide_id=slide_id, mag=mag)
            print(f"Found {len(patch_coords)} useful patches for slide {slide_id}")

            if len(patch_coords) == 0:
                raise ValueError(f"No valid patch found in slide {slide_id}")

            if not args.only_coords:
                process_slide(args, slide_path, patch_coords)
                # don't visualize
                visualize_patches(args, slide_path, target_mag=max(args.magnifications))
            else:
                store_available_coords(args, slide_id, patch_coords, mag=mag)

            with open(f"{args.patch_folder}/success.txt", "a") as f:
                f.write(f"{slide_id}\n")
            # delete entry in the fail file if it exists
            with open(f"{args.patch_folder}/fail.txt", "r") as f:
                lines = f.readlines()
            with open(f"{args.patch_folder}/fail.txt", "w") as f:
                for line in lines:
                    if slide_id not in line:
                        f.write(line)
        # if False:
        except Exception as e:
            print(f"Failed to process {slide_id}: {e}")
            if str(e) == "Unsupported or missing image file":
                with open(f"{args.patch_folder}/files_corruption.txt", "a") as f:
                    f.write(f"{slide_id}\n")
            with open(f"{args.patch_folder}/fail.txt", "a") as f:
                f.write(f"{slide_id}\n")


if __name__ == "__main__":
    main()

# python create_tiles_bao.py \
#     --csv_path /home/bal753/DFCI_immune_file_manual.csv \
#     --wsi_col FILE_PATH \
#     --patch_folder /n/scratch/users/b/bal753/DFCI_EN \
#     --patch_size 224 \
#     --stride 224 \
#     --output_size 224 \
#     --tissue_threshold 5 \
#     --magnifications 20\
#     --n_workers 16 --only_coords