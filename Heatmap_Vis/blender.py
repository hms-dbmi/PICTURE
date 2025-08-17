

import os
import sys
import scipy
import math
# from jsonc_parser.parser import JsoncParser
from PIL import Image, ImageFilter
from tqdm import tqdm
from skimage import data
from scipy import ndimage
from skimage.color import rgb2gray
from argparse import ArgumentParser, Namespace
from skimage import feature
import skimage
import cv2
import numpy as np

from matplotlib import pyplot as plt


def rgbHueSat(pix):
    # pixImg=Image.fromarray(pix)
    pixHsv = pix.convert('HSV')
    pixHsvArray = np.array(pixHsv, dtype=np.float32)
    return pixHsvArray[:, :, 0],  pixHsvArray[:, :, 1]


def OpticalDensityThreshold(I, Io=240, beta=0.15):
    # calculate optical density
    OD = -np.log((I.astype(np.float32)+1)/Io)
    # remove transparent pixels
    ODhat = ~np.any(OD < beta, axis=2).astype(np.uint8)
    return OD, ODhat


class HeatmapBlender:
    def __init__(self, base_map, heat_map, mask_map=None):
        if isinstance(base_map, str):
            base_map = cv2.imread(base_map)
        if isinstance(heat_map, str):
            heat_map = cv2.imread(heat_map)
        if isinstance(mask_map, str):
            mask_map = cv2.imread(mask_map)

        self.base_map = base_map
        self.heat_map = heat_map
        if mask_map is None:
            self.background_mask = mask_map
        else:
            self.background_mask = 1 - mask_map

    def gaussian_blur_image(self, image, ksize, sigmaX):
        # image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blurred_image = cv2.GaussianBlur(image, ksize, sigmaX)
        return blurred_image

    def resize_image(self, image, target_shape):
        resized_image = cv2.resize(image, (target_shape[1], target_shape[0]))
        return resized_image

    def blend_images(self, base_image, overlay_image, alpha):
        blended_image = cv2.addWeighted(
            base_image, 1 - alpha, overlay_image, alpha, 0)
        return blended_image

    def select_background(self, image, seed_points):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        background_mask = np.zeros(gray_image.shape, dtype=np.uint8)

        for seed_point in seed_points:
            mask = np.zeros(
                (gray_image.shape[0] + 2, gray_image.shape[1] + 2), dtype=np.uint8)
            cv2.floodFill(gray_image, mask, seed_point, 255, (self.tolerance,)
                          * 3, (self.tolerance,) * 3, cv2.FLOODFILL_MASK_ONLY)
            mask = mask[1:-1, 1:-1].copy()
            background_mask = cv2.bitwise_or(background_mask, mask)

        return background_mask

    def rgbHueSat(self, pix):
        # pixImg=Image.fromarray(pix)

        pixHsv = pix.convert('HSV')
        pixHsvArray = np.array(pixHsv, dtype=np.float32)

        return pixHsvArray[:, :, 0],  pixHsvArray[:, :, 1]

    def OpticalDensityThreshold(self, I, Io=240, beta=0.15):
        # calculate optical density
        OD = -np.log((I.astype(np.float32)+1)/Io)

        # remove transparent pixels
        ODhat = ~np.any(OD < beta, axis=2)

        return OD, ODhat

    def get_mask(self,
                 infile,
                 params={
                     # size for median filter (for filtering out holes)
                     "filter_size": 13,
                     # params for optical density thresholds (see OpticalDensityThreshold),
                     "OD_Io": 250,
                     "OD_beta": 0.00,
                     # Hue thresholds (in degree [0~360])
                     "hueLowerBound": 70.0,
                     "hueUpperBound": 230.0,
                     # Saturation thresholds (range [0,1])
                     "saturationLowerBound": 0.1,
                     # thresholds for black areas (range [0,255])
                     "BlackThresold": 10,
                     # LoG kernel width (DISABLED)
                     "LOGsigma": 3,
                     # LoG threshold (DISABLED)
                     "LOGLowerBound": 0.000
                 }):
        params = Namespace(**params)

        ##### Read params from Json file ########
        # locals().update(JsoncParser.parse_file(params))
        # read params from JSON file
        print(params)
        params.hueLowerBound = params.hueLowerBound/360*255
        params.hueUpperBound = params.hueUpperBound/360*255
        params.saturationLowerBound = params.saturationLowerBound * 255
        ###########################

        # print(filename)
        I = Image.fromarray(infile)
        I = I.filter(ImageFilter.MedianFilter(
            size=params.filter_size))  # median filter
        pix = np.array(I)
        pix = pix[:, :, 0:3]

        pixR = pix[:, :, 0]
        pixG = pix[:, :, 1]
        pixB = pix[:, :, 2]
        pixRange = np.ptp(pix, axis=2)
        # nonEmpty = (pixRange > nonEmptyRangeThreshold)
        # nonEmptySum = nonEmpty.sum()
        OD, nonEmpty = self.OpticalDensityThreshold(
            pix, Io=params.OD_Io, beta=params.OD_beta)
        OD_gray = rgb2gray(OD)
        OD_LoG = ndimage.gaussian_laplace(OD_gray, sigma=params.LOGsigma)
        OD_abs_LoG = np.abs(OD_LoG)

        nonEmptySum = nonEmpty.sum()
        pixHue, pixSat = self.rgbHueSat(I)
        nonEmpty = np.where(nonEmpty, 1, 0)
        nonBlack = np.where(np.min(pix, axis=2) > params.BlackThresold, 1, 0)
        nonHue = np.where(np.logical_or(
            pixHue < params.hueLowerBound, pixHue > params.hueUpperBound), 1, 0)
        nonLoG = np.where(OD_abs_LoG > params.LOGLowerBound, 1, 0)
        # nonLoG = np.where(OD_LoG < -LOGLowerBound,1,0)
        nonSat = np.where(pixSat > params.saturationLowerBound, 1, 0)
        nonAllCriteria = nonEmpty * nonHue * nonSat * nonBlack * nonLoG
        nonAllCriteria = nonAllCriteria.astype(np.uint8)
        # nonAllCriteria_edge = ndimage.gaussian_laplace(nonAllCriteria, sigma=LOGsigma)
        nonAllCriteria_edge = feature.canny(
            nonAllCriteria.astype(np.float32), sigma=3)
        nonAllCriteria_rgb = np.stack([nonAllCriteria]*3, axis=2)*255

        pixR_edge = pixR
        pixG_edge = pixG
        pixB_edge = pixB
        pixR_edge[nonAllCriteria_edge == True] = 255
        pixG_edge[nonAllCriteria_edge == True] = 255
        pixB_edge[nonAllCriteria_edge == True] = 0
        pix_edge = np.stack([pixR_edge, pixG_edge, pixB_edge], axis=2)

        if nonAllCriteria[0][0] == 0:
            nonAllCriteria = 1 - nonAllCriteria
        return nonAllCriteria

    def get_final_image(self, ksize=(15, 15), sigmaX=None, alpha=0.3, tolerance=1, save=False):
        self.tolerance = tolerance
        base_map = self.base_map
        base_map = cv2.cvtColor(base_map, cv2.COLOR_BGR2RGB)

        if sigmaX is None:
            blurred_heatmap = self.heat_map
        else:
            blurred_heatmap = self.gaussian_blur_image(
                self.heat_map, ksize, sigmaX)
        resized_heatmap = self.resize_image(
            blurred_heatmap, base_map.shape[:2])

        blended_image = self.blend_images(base_map, resized_heatmap, alpha)

        # Get the seed points for the corners
        seed_points = [
            (0, 0),  # Top-left
            (base_map.shape[1] - 1, 0),  # Top-right
            (0, base_map.shape[0] - 1),  # Bottom-left
            (base_map.shape[1] - 1, base_map.shape[0] - 1),  # Bottom-right
        ]

#         self.background_mask = self.select_background(base_map, seed_points)
        if self.background_mask is None:
            self.background_mask = self.get_mask(base_map)
        A_bool = self.background_mask.astype(bool)
        white = np.full(
            (self.background_mask.shape[0], self.background_mask.shape[1], 3), 255)
        blended_image[A_bool] = white[A_bool]

        # if save:
        #     save_dir = self.base_map_path.replace('.png', 'synthetic_HEAT.png')
        #     # Convert the ndarray to a PIL Image object.
        #     image = Image.fromarray(blended_image)

        #     # Save the PIL Image object to a file in PNG format.
        #     image.save(save_dir)

        return blended_image


def blend_heatmap(base_map_path, heatmap_path, outpath):
    
    blender = HeatmapBlender(base_map_path, heatmap_path)
    final_image = blender.get_final_image(
        ksize=(151, 151), sigmaX=100.0, tolerance=10, save=True)

    print('FINAL IMAGE SHAPE:', final_image.shape)

    # Get the original images for plotting
    original_image = cv2.imread(heatmap_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    base_map = cv2.imread(base_map_path)
    base_map = cv2.cvtColor(base_map, cv2.COLOR_BGR2RGB)

    # Get the background mask
    seed_points = [
        (0, 0),  # Top-left
        (original_image.shape[1] - 1, 0),  # Top-right
        (0, original_image.shape[0] - 1),  # Bottom-left
        (original_image.shape[1] - 1,
         original_image.shape[0] - 1),  # Bottom-right
    ]
    background_mask = blender.background_mask

    # Plot the images
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs[0, 0].imshow(original_image)
    axs[0, 0].set_title("Original Image")
    axs[0, 1].imshow(base_map)
    axs[0, 1].set_title("Base Map")
    axs[1, 0].imshow(background_mask, cmap='gray')
    axs[1, 0].set_title("Background Mask")
    axs[1, 1].imshow(final_image)
    axs[1, 1].set_title("Final Image")

    # Remove the axes for a cleaner look
    for ax in axs.ravel():
        ax.axis('off')
    outdir = os.path.dirname(outpath)
    os.makedirs(outdir, exist_ok=True)
    # plt.show()
    plt.savefig(outpath)
