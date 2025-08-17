
import argparse
from blender import *
from matplotlib.colors import ListedColormap
from matplotlib import pylab as plt
from skimage.transform import resize
from tempfile import mkdtemp
import girder_client
import json
import pickle
import h5py
import colorsys
import csv
import scipy.misc
import openslide
import imageio
import zipfile
import requests
import pandas as pd
from typing import Dict, Any
import timm
from timm.models.layers.helpers import to_2tuple
from blender import HeatmapBlender

from PIL import Image, ImageFilter, ImageOps
import PIL
import glob
import sys
import os
from collections import OrderedDict
from skimage.color import rgb2hsv, rgb2gray

import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
import torchvision
from torchvision import datasets, models, transforms
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import shutil
import gc
import time
import random
from datetime import datetime
from skimage import exposure
# from tqdm.notebook import tqdm
from sklearn import model_selection, metrics
'''
Heatmap generation. By Junhan.Zhao, Ph.D. and Shih-Yen.Lin, Ph.D. @HMS-YuLab 
'''

from model_loaders import *
import scipy
import math
# from jsonc_parser.parser import JsoncParser
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


import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.functional import kl_div, softmax, log_softmax
import matplotlib.pyplot as plt
import time

import copy
import matplotlib.pyplot as plt
from macenko_mod import TorchMacenkoNormalizer

sys.path.insert(0, '../..')


# imagenet mean std

rgb_mean = (0.485, 0.456, 0.406)
rgb_std = (0.229, 0.224, 0.225)

transform_normal = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def MaskedNormalizeData(data):
    minval = np.min(data[data > 0])
    data = (data - minval) / (np.max(data) - minval)
    return np.maximum(data, 0)


def ctranspath():
    # model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)

    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
    model.patch_embed = ConvStem(
        img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm, flatten=True)
    # patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    # img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
    # norm_layer=norm_layer if self.patch_norm else None)
    return model


def predict(net=None, loader=None, a_device=None, no_of_classes=5, a_image=None):
    net.eval()

    if loader != None and a_image == None:
        itertor = iter(loader)
        total_step = len(loader)
        print("TOTAL STEPS: {}".format(total_step))
        predictions_all = []
        label_all = []
        running_corrects = 0

        total_dict = {i: 0 for i in range(no_of_classes)}
        hit_dict = {i: 0 for i in range(no_of_classes)}
        pbar = tqdm(range(total_step), mininterval=60)
        pbar.set_description("Making predictions...")
        for step in pbar:
            with torch.no_grad():
                imgs, labels, locs = next(itertor)
                imgs = imgs.to(a_device)
                labels = labels.to(device=a_device, dtype=torch.int64)

                logps = net(imgs)

                total_dict = {
                    i: total_dict[i] + labels.tolist().count(i) for i in range(no_of_classes)}

                # Update running statistics
                probs = torch.nn.functional.softmax(
                    logps, dim=1)  # probabilities

                # Running count of correctly identified classes
                ps = torch.exp(logps)
                _, predictions = ps.topk(1, dim=1)  # top predictions
                equals = predictions == labels.view(*predictions.shape)

                if len(predictions_all) == 0:
                    predictions_all = ps.detach().cpu().numpy()
                    label_all = labels.tolist()
                    probs_all = probs.detach().cpu().numpy()
                else:
                    predictions_all = np.vstack(
                        (predictions_all, ps.detach().cpu().numpy()))
                    probs_all = np.vstack(
                        (probs_all, probs.detach().cpu().numpy()))
                    label_all.extend(labels.tolist())

                # get all T/F indices
                all_hits = equals.view(equals.shape[0]).tolist()
                print(locs)
                print(np.asarray(locs)[np.where(all_hits)[0]])
                all_corrects = labels[all_hits]

                hit_dict = {
                    i: hit_dict[i] + all_corrects.tolist().count(i) for i in range(no_of_classes)}
                running_corrects += torch.sum(
                    equals.type(torch.FloatTensor)).item()

        phase_acc = running_corrects / sum(total_dict.values())

        y_true = label_all.copy()
        y_pred = np.argmax(predictions_all, axis=1)
        # print("accuray:", len(np.arange(len(y_true))[y_true == y_pred]) / len(y_true))
        metrics.confusion_matrix(y_true, y_pred)
        # print(metrics.classification_report(y_true, y_pred, digits=3))
        #
        # print("\n".join("class: {}\t counts: {}".format(k, v) for k, v in total_dict.items()))
        # print("\n".join("class: {}\t counts: {}".format(k, v) for k, v in hit_dict.items()))

    elif (a_image != None) and loader == None:
        with torch.no_grad():
            img = transformations(a_image)
            img = img.unsqueeze(0)
            img.to(device)
            # forward propagation
            logps = net(img)

            # Update running statistics
            probs = torch.nn.functional.softmax(logps, dim=1)  # probabilities
            # Running count of correctly identified classes
            ps = torch.exp(logps)
            _, predictions = ps.topk(1, dim=1)  # top predictions
            # print('Prediction is:', predictions.tolist()[0][0])

    elif (a_image != None) and (loader != None):
        output = []
        path_lst = []
        hue = []
        saturation = []
        with torch.no_grad():
            itertor = iter(loader)
            total_step = len(loader)
            # for step in range(total_step):

            pbar = tqdm(range(total_step), mininterval=60)
            pbar.set_description("Making predictions...")
            for step in pbar:
                # print(f'{step}/ {total_step}')
                imgs, paths = next(itertor)
                pixs = np.asarray(imgs)
                pixs = pixs[:, :, :, 0:3]
                for pix in pixs:
                    hsv_img = rgb2hsv(pix)
                    H, S, V = np.mean(hsv_img[:, :, 0]), np.mean(
                        hsv_img[:, :, 1]), np.mean(hsv_img[:, :, 2])
                    hue.append(H)
                    saturation.append(S)

                imgs = imgs.to(a_device)
                # forward propagation
                logps = net(imgs)
                # Update running statistics
                probs = torch.nn.functional.softmax(
                    logps, dim=1)  # probabilities
                # Running count of correctly identified classes
                probs = probs.cpu().detach().numpy()
                output.append(list(probs))
                path_lst.extend(list(paths))

                ps = torch.exp(logps)
                _, predictions = ps.topk(1, dim=1)  # top predictions
        #             print('Prediction is:' , predictions.tolist()[0][0])
        output = np.vstack(output)
        if no_of_classes == 2:
            df = pd.DataFrame(output, columns=['Low', 'High'])
        elif no_of_classes == 3:
            df = pd.DataFrame(output, columns=['Low', 'Med', 'High'])

        df['image_path'] = path_lst
        df['S'] = saturation
        df['H'] = hue
    return df


def mapped(x, count, value):
    """ 
    Calculate the averaged score map
    x : current score map 
    count: count map
    value: fill value
    """
    new_count = count + 1
    old_ratio = count / new_count
    new_ratio = 1 / new_count
    y = old_ratio * x + new_ratio * value

    return y, new_count


def save_h5(tile_save_path, hdf5_path, clean=True):

    img_list = glob.glob(f"{tile_save_path}/*jpg*")
    import tables
    img_dtype = tables.UInt8Atom()
    data_shape = (0, 224, 224, 3)

    print("Creating earrays")
    # open the specified hdf5 file and create earrays to store the images
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    train_storage = hdf5_file.create_earray(
        hdf5_file.root, 'imgs', img_dtype, shape=data_shape)
    hdf5_file.create_array(hdf5_file.root, 'img_paths', img_list)

    print("reading images into storage")
    # read training images into train_storage
    for i in range(len(img_list)):
        if i % 1000 == 0 and i > 1:
            print('Processed training data: {}/{}'.format(i, len(img_list)))
        addr = img_list[i]
        img = Image.open(addr)
        img = img.resize((224, 224), resample=Image.BICUBIC)
        img_arr = np.asarray(img)
        # Put None here to add an extra dimension [None,224,224,3]
        train_storage.append(img_arr[None])

    # Good practice
    hdf5_file.close()

    if clean:
        try:
            shutil.rmtree(tile_save_path)
            print("Image folder deleted")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


class HDF5Dataset(Dataset):

    def __init__(self, path):
        self.file_path = path
        self.dataset = None
        file = h5py.File(path, "r")
        self.dataset_len = len(file['imgs'])
        self.transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)])

    def __getitem__(self, index):
        if self.dataset is None:
            self.imgs = h5py.File(self.file_path, 'r')['imgs']
            self.paths = h5py.File(self.file_path, 'r')['img_paths']
            cur_img = self.imgs[index]
            path = self.paths[index].decode('UTF-8')
            PIL_image = Image.fromarray(np.uint8(cur_img)).convert('RGB')
            img = self.transformations(PIL_image)

        return (img, path)

    def __len__(self):
        return self.dataset_len


def bounding_box_naive(points):
    """returns a list containing the bottom left and the top right
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)

    return bot_left_x, bot_left_y, top_right_x, top_right_y


def get_mask(
        I,
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
    ##### Read params from Json file ########
    # locals().update(JsoncParser.parse_file(params))
    # read params from JSON file
    params = Namespace(**params)

    # print(params)
    params.hueLowerBound = params.hueLowerBound/360*255
    params.hueUpperBound = params.hueUpperBound/360*255
    params.saturationLowerBound = params.saturationLowerBound * 255
    ###########################

    # print(filename)
    # I = Image.fromarray(infile)
    I = I.filter(ImageFilter.MedianFilter(
        size=params.filter_size))  # median filter
    pixHue, pixSat = rgbHueSat(I)
    pix = np.array(I)
    del I
    pix = pix[:, :, 0:3]
    #not transparent
    _, nonAllCriteria = OpticalDensityThreshold(
        pix, Io=params.OD_Io, beta=params.OD_beta)

    #not black
    nonAllCriteria = nonAllCriteria * np.where(np.min(pix, axis=2) >
                                               params.BlackThresold, 1, 0).astype(np.uint8)
    # Hue
    nonAllCriteria = nonAllCriteria * np.where(np.logical_or(
        pixHue < params.hueLowerBound, pixHue > params.hueUpperBound), 1, 0).astype(np.uint8)
    # Saturation
    nonAllCriteria = nonAllCriteria * np.where(pixSat > params.saturationLowerBound,
                                               1, 0).astype(np.uint8)

    return nonAllCriteria


def get_sampling_params(slide, mag_power=20):
    # Get the optimal openslide level and subsample rate, given a magnification power
    mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    assert mag >= mag_power, f"Magnification of the slide ({mag}X) is smaller than the desired magnification ({mag_power}X)."
    ds_rate = mag / mag_power

    lvl_ds_rates = np.array(slide.level_downsamples).astype(np.int32)
    levels = np.arange(len(lvl_ds_rates))
    # Get levels that is larger than the given mag power
    idx_larger = np.argwhere(lvl_ds_rates <= ds_rate).flatten()
    lvl_ds_rates = lvl_ds_rates[idx_larger]
    levels = levels[idx_larger]
    # get the closest & larger mag power
    idx = np.argmax(lvl_ds_rates)
    closest_ds_rate = lvl_ds_rates[idx]
    opt_level = levels[idx]
    opt_ds_rate = ds_rate / closest_ds_rate

    return opt_level, opt_ds_rate


def read_region_by_power(slide, start, mag_power, width):
    opt_level, opt_ds_rate = get_sampling_params(slide, mag_power)
    read_width = tuple([int(opt_ds_rate*x) for x in width])
    im1 = slide.read_region(start, opt_level, read_width)
    if opt_ds_rate != 1:
        im1 = im1.resize(width, resample=Image.LINEAR)
    return im1


class Create_Heatmap:
    """ class for heatmap creation """

    def __init__(self, slide_path,
                 xStride=250, yStride=250, xPatch=1000, yPatch=1000, mag_power=20,
                 color_norm=False, Io_source=250, Io_target=250, beta=0.15,
                 thumbnail_size=1000):
        """
        slide_path: Path to the WSI image
        xStride, yStride: stride for the tiles (pixels)
        xPatch, yPatch: width for the tiles (pixels)
        level: level of magnification
        color_norm: Use color normalization
        thumbnail_size: maximum size of the thumbnail
        """
        self.slide_id = slide_path.split('/')[-1]
        self.mr_image = openslide.OpenSlide(slide_path)
        self.mag_power = mag_power
        # level, ds_rate = get_sampling_params(self.mr_image, mag_power)
        mag_base = int(
            self.mr_image.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        ds_rate = mag_base/mag_power
        # self.level = level
        # self.ds_rate = ds_rate
        self.color_norm = color_norm
        self.Io_source = Io_source
        self.Io_target = Io_target
        self.beta = beta
        self.orig_xDim = self.mr_image.level_dimensions[0][0]//ds_rate
        self.orig_yDim = self.mr_image.level_dimensions[0][1]//ds_rate
        self.xStride, self.yStride, self.xPatch, self.yPatch = xStride, yStride, xPatch, yPatch
        self.thumbnail_size = thumbnail_size
        self.metric = {'orig_xDim': self.orig_xDim, 'orig_yDim': self.orig_yDim, 'xStride': self.xStride,
                       'yStride': self.yStride, 'xPatch': self.xPatch, 'yPatch': self.yPatch, "slide_id": self.slide_id}

    def get_bounding_box(self,
                         border_width=1000,
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
        if not hasattr(self, "thumbnail"):
            self.initialize_image_mask(params)
        # thumbnail = self.mr_image.get_thumbnail((self.thumbnail_size,thumbnail_size))
        # mask = self.get_mask(thumbnail,params)

        xDim = self.mr_image.level_dimensions[0][0]
        yDim = self.mr_image.level_dimensions[0][1]

        squeeze_x_thumb = np.argwhere(np.max(self.mask, axis=0)).flatten()
        squeeze_y_thumb = np.argwhere(np.max(self.mask, axis=1)).flatten()
        bb_x_thumb = np.array(
            [np.min(squeeze_x_thumb), np.max(squeeze_x_thumb)])
        bb_y_thumb = np.array(
            [np.min(squeeze_y_thumb), np.max(squeeze_y_thumb)])

        ds_rate = np.array([xDim, yDim])/self.thumbnail.size
        bb_x = np.round(bb_x_thumb * ds_rate[0])
        bb_y = np.round(bb_y_thumb * ds_rate[1])
        bb_x[0] = np.maximum(bb_x[0] - border_width, 0)
        bb_y[0] = np.maximum(bb_y[0] - border_width, 0)
        bb_x[1] = np.minimum(bb_x[1] + border_width, xDim-1)
        bb_y[1] = np.minimum(bb_y[1] + border_width, yDim-1)
        # row = {'labels': args.label,'bottom_left_X': bl_X, 'bottom_left_Y': bl_Y , 'top_right_X': tr_X, 'top_right_Y':tr_Y}
        bl_X = int(bb_x[0])
        bl_Y = int(bb_y[0])
        tr_X = int(bb_x[1])
        tr_Y = int(bb_y[1])

        # out_lvl=4
        # ds = self.mr_image.level_downsamples[out_lvl]
        # read_size = np.array([tr_X,tr_Y])//ds
        # read_image = self.mr_image.read_region((bl_X,bl_Y), out_lvl,read_size.astype(np.int32))

        # read_image.convert('RGB').save('test.jpg')

        return np.array([bl_X, bl_Y, tr_X, tr_Y])

    def initialize_image_mask(self,
                              params={
            # size for median filter (for filtering out holes)
            "filter_size": 13,
            # params for optical density thresholds (see OpticalDensityThreshold),
            "OD_Io": 250,
            "OD_beta": 0.00,
            "hueLowerBound": 70.0,  # Hue thresholds (in degree [0~360])
            "hueUpperBound": 230.0,
            "saturationLowerBound": 0.1,  # Saturation thresholds (range [0,1])
            # thresholds for black areas (range [0,255])
            "BlackThresold": 10,
            "LOGsigma": 3,                      # LoG kernel width (DISABLED)
            "LOGLowerBound": 0.000              # LoG threshold (DISABLED)
                                  }):
        thumbnail = self.mr_image.get_thumbnail(
            (self.thumbnail_size, self.thumbnail_size))
        mask = get_mask(thumbnail, params)
        self.thumbnail = thumbnail
        self.mask = Image.fromarray(mask)

    def if_tile_nonempty(self, i, j, tissue_ratio_threshold=0.1):
        xDim = self.mr_image.level_dimensions[0][0]
        yDim = self.mr_image.level_dimensions[0][1]
        read_X = self.xStride * i
        read_Y = self.yStride * j

        ds_rate_lvl = np.array(
            [self.orig_xDim, self.orig_yDim])/self.thumbnail.size
        ds_rate_0 = np.array([xDim, yDim])/self.thumbnail.size

        startCoord = np.array([self.startX + read_X, self.startY + read_Y])
        imgWidth = np.array([self.xPatch, self.yPatch])
        startCoord_ds = startCoord / ds_rate_0
        imgWidth_ds = imgWidth / ds_rate_lvl
        endCoord_ds = startCoord_ds + imgWidth_ds

        ds_mask = self.mask.crop(
            (startCoord_ds[0], startCoord_ds[1], endCoord_ds[0], endCoord_ds[1]))
        mask_np = np.array(ds_mask)
        tissue_ratio = np.sum(mask_np)/mask_np.size
        if tissue_ratio < tissue_ratio_threshold:
            # if np.max(ds_mask) == 0:
            return False
        else:
            return True

    def tile_patch(self, tile_save_path, a_row):
        self.row = a_row

        if not hasattr(self, "thumbnail"):
            self.initialize_image_mask()

        start_time = time.time()
        # create a folder for saving tiles
        if not os.path.exists(tile_save_path):
            os.makedirs(tile_save_path)

        # save the input to instance
        self.tile_save_path = tile_save_path
        self.startX, self.startY, self.extentX, self.extentY = a_row['bottom_left_X'], a_row['bottom_left_Y'], (
            a_row['top_right_X'] - a_row['bottom_left_X']), (a_row['top_right_Y'] - a_row['bottom_left_Y'])

        self.save_key = f"{self.slide_id}+{a_row['labels']}+{a_row['bottom_left_X']}+{a_row['bottom_left_Y']}"

        nonEmptyTally = []

        for i in range(int(self.extentX / self.xStride)):
            for j in range(int(self.extentY / self.yStride)):
                if self.if_tile_nonempty(i, j):
                    nonEmptyTally.append([int(i), int(j)])

        sortedNonEmptyTally = nonEmptyTally
        print("Total Patches Max: {}".format(len(sortedNonEmptyTally)))
        y_uniques = self.startY + self.yStride * \
            np.unique(np.array(sortedNonEmptyTally)[:, 1])
        nStart = 0
        pbar = tqdm(range(nStart, len(sortedNonEmptyTally)),
                    mininterval=60)
        pbar.set_description("Writing Image Tiles...")
        thumbnail = np.array(self.thumbnail)
        xDim = self.mr_image.level_dimensions[0][0]
        yDim = self.mr_image.level_dimensions[0][1]
        if self.color_norm:
            torch_normalizer = TorchMacenkoNormalizer()
            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x*255)
            ])

        for i in pbar:
            read_X = self.xStride * sortedNonEmptyTally[i][0]
            read_Y = self.yStride * sortedNonEmptyTally[i][1]

            startCoord = np.array([self.startX + read_X, self.startY + read_Y])
            imgWidth = np.array([self.xPatch, self.yPatch])

            # image_patch = self.mr_image.read_region(
            #     startCoord, self.level, imgWidth_read)
            # image_patch = image_patch.resize(imgWidth, resample=Image.LINEAR)

            image_patch = read_region_by_power(
                self.mr_image, startCoord, self.mag_power, imgWidth)

            if self.color_norm:
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                t_source = T(image_patch).to(device)
                t_source = t_source[:3, :, :]
                try:
                    norm, _, _ = torch_normalizer.normalize(
                        I=t_source, stains=False, Io=self.Io_source, Io_out=self.Io_target, beta=self.beta)
                except:
                    print(f"Failed normalizing tile {i}. Skipping...")
                    continue
                image_patch = Image.fromarray(
                    norm.cpu().numpy().astype(np.uint8))

            # pix = np.array(image_patch)
            outputFileName = self.tile_save_path + "/_" + \
                str(self.startX + read_X) + "_" + \
                str(self.startY + read_Y) + ".jpg"
            # im = Image.fromarray(pix)
            im = image_patch.convert("RGB")
            im.save(outputFileName)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("all patches save done")


def create_heatmap(hdf5_path, model, hm_save_folder, metrics, a_row, slide_path,
                   save_crop=True, col='Low', no_of_classes=2, csv=None, device_id='0',
                   normalize=False, tile_mag_power=20, wsi_mag_power=1, tissue_ratio_threshold=0.1,
                   predict_only=False, heatmap_only=False, alpha=0.3,
                   equalize=False):
    '''
    create_heatmap
    '''
    # calculate some parameters
    mr_image = openslide.OpenSlide(slide_path)

    slide_id = metrics["slide_id"]
    save_key = f"{slide_id}+{a_row['labels']}+{col}+{a_row['bottom_left_X']}+{a_row['bottom_left_Y']}"
    # tile_ds_rate = mr_image.level_downsamples[tile_mag_level]
    # wsi_ds_rate = mr_image.level_downsamples[wsi_mag_level]

    print(f'Magnification power for tile prediction: {tile_mag_power}')
    print(f'Magnification power for WSI heatmap generation: {wsi_mag_power}')
    print(f'Magnification power for WSI heatmap generation: {wsi_mag_power}')

    obj_mag_power = int(
        mr_image.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    tile_ds_rate = obj_mag_power / tile_mag_power
    wsi_ds_rate = obj_mag_power / wsi_mag_power

    level, ds_rate = get_sampling_params(mr_image, tile_mag_power)
    print(
        f'Optimal sampling params for tile prediction: level={level}, ds_rate={ds_rate}')
    read_level, read_ds_rate = get_sampling_params(mr_image, wsi_mag_power)
    print(
        f'Optimal sampling params for tile prediction: level={level}, ds_rate={ds_rate}')

    startX, startY, extentX, extentY = a_row['bottom_left_X'], a_row['bottom_left_Y'], int(
        (a_row['top_right_X'] - a_row['bottom_left_X'])/tile_ds_rate), int((a_row['top_right_Y'] - a_row['bottom_left_Y'])/tile_ds_rate)

    extentX_wsi = int(extentX / (wsi_ds_rate/tile_ds_rate))
    extentY_wsi = int(extentY / (wsi_ds_rate/tile_ds_rate))
    startX_wsi = int(startX / wsi_ds_rate)
    startY_wsi = int(startY / wsi_ds_rate)
    xPatch_ds = int(metrics['xPatch'] / (wsi_ds_rate/tile_ds_rate))
    yPatch_ds = int(metrics['yPatch'] / (wsi_ds_rate/tile_ds_rate))

    ###
    thumbnail = mr_image.get_thumbnail((1000, 1000))
    thumbnail_path = os.path.join(hm_save_folder, f'{save_key}_thumbnail.png')
    thumbnail.save(thumbnail_path)

    # auto_csv = hdf5_path.replace('.hdf5', '.csv')

    if os.path.isfile(csv):
        if predict_only:
            return {}
        print(f'Read tf preds from automatically saved pickle {csv}')
        output = pd.read_pickle(csv)
    elif heatmap_only:
        raise RuntimeError(
            f'pickle file {csv} not found, but heatmap_only is also set to true. consider running inference with heatmap_only=False first.')
    else:
        print(f'GPU:{device_id} is in use')
        # MODEL_PATH = model_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print('model is loaded!')
        from torch.utils.data import Dataset, DataLoader
        test_loader = DataLoader(HDF5Dataset(
            path=hdf5_path), batch_size=400, shuffle=False, drop_last=False)
        output = predict(net=model, loader=test_loader,
                         a_device=device, a_image=1, no_of_classes=no_of_classes)
        if csv is not None:
            print(f'Writing tf preds to pickle {csv}')
            output.to_pickle(csv)
    print('predicting done')
    if predict_only:
        return {}
    print('working on visualizations...')

    output['locs'] = output['image_path'].apply(
        lambda x: [x.split('/')[-1].split('_')[1], x.split('/')[-1].split('_')[2].split('.')[0]])

    saturas = output['S'].to_numpy()
    cutoff_sat_idx = np.argwhere(saturas < 0.1)

    weights = output[col].to_numpy()
    # overwrite the white tile probs
    # weights[cutoff_sat_idx] = 0

    norm_weights = [float(i) / max(weights) for i in weights]

    score_map = np.zeros((extentX_wsi, extentY_wsi))
    score_map_count = np.zeros_like(score_map)
    print("reading image...")
    # im1 = mr_image.read_region(
    #     (startX, startY), read_level, read_size)
    im1 = read_region_by_power(
        mr_image, (startX, startY), wsi_mag_power, (extentX_wsi, extentY_wsi))

    if save_crop:
        crop_path = os.path.join(hm_save_folder, f'{save_key}_crop.png')
        im1.save(crop_path, format="png", optimize=True)
        print("{} annotated crop saved".format(save_key))
    else:
        crop_path = None
    print("Estimating mask...")
    im_mask = get_mask(im1)
    # delete the loaded im1 for now (in case there is not enough memory for large images)
    print(im_mask.shape)

    del im1

    pbar = tqdm(enumerate(output['locs'].tolist()),
                mininterval=60)
    pbar.set_description("mapping tiles...")
    for i, loc in pbar:
        # if i % 100 == 0:
        #     print(f'mapping {i}th tile...')
        value = norm_weights[i]
        # if i % 100 == 0:
        # print(int(loc[0]),int(loc[1]),value)

        l0 = int(int(loc[0])/wsi_ds_rate) - startX_wsi
        l1 = int(int(loc[1])/wsi_ds_rate) - startY_wsi
        mask_patch = im_mask[l1:(l1 + yPatch_ds), l0:(l0 + xPatch_ds)]
        tissue_ratio = np.sum(mask_patch)/mask_patch.size
        if tissue_ratio > tissue_ratio_threshold:
            score_map[l0:(l0 + xPatch_ds), l1:(l1 + yPatch_ds)], score_map_count[l0:(l0 + xPatch_ds), l1:(l1 + yPatch_ds)] = mapped(
                score_map[l0:(l0 + xPatch_ds), l1:(l1 + yPatch_ds)], score_map_count[l0:(l0 + xPatch_ds), l1:(l1 + yPatch_ds)], value)

    # mask = np.transpose(mask)
    print("Score Map Done")
    print(metrics)
    print(score_map.shape)
    # score_map = score_map[startX_wsi:startX_wsi + extentX_wsi,
    #                       startY_wsi:startY_wsi + extentY_wsi]
    score_map = np.transpose(score_map)
    # new_mask = np.flip(new_mask)

    # calibration: the white-ish tiles normally have very high probability to be unlike class (e.g., IDH-wt).
    # therefore, we need to remove those high-probs by finding these white regions using saturation

    # score_map = cv2.resize(mask, newsize)
    # del mask
    score_map = score_map * im_mask
    # score_map = NormalizeData(score_map)

    if equalize:  # Do histogram normalization
        # score_map = ImageOps.equalize(score_map)

        # # Adaptive Equalization
        # idx_nz = score_map > 0
        # cam_nz = score_map[idx_nz]
        # cam_nz = exposure.equalize_adapthist(cam_nz, clip_limit=1e-3)
        # score_map[idx_nz] = cam_nz

        score_map = exposure.equalize_hist(score_map, 1000, im_mask)

    elif normalize:
        score_map = MaskedNormalizeData(score_map)

    print(np.max(score_map))
    import seaborn as sns
    from matplotlib.pyplot import show
    import matplotlib

    # sns.set(rc={'figure.figsize': (100, 100 * extentY/extentX)})

    # from scipy import ndimage
    # sigma_y = 3
    # sigma_x = 3
    # sigma = [sigma_y, sigma_x]
    # if normalize:
    #     print("Normalize the weights")
    #     grayscale_cam_blur = ndimage.filters.gaussian_filter(
    #         score_map, sigma)

    # else:
    #     grayscale_cam_blur = ndimage.filters.gaussian_filter(
    #         score_map, sigma)

    # # Masking and normalizing final heatmap
    # grayscale_cam_blur = grayscale_cam_blur * im_mask
    # grayscale_cam_blur = grayscale_cam_blur / np.max(grayscale_cam_blur)
    # ##

    color_maps = {
        'JET': cv2.COLORMAP_JET,
        # 'BONE': cv2.COLORMAP_BONE,
        # 'HOT': cv2.COLORMAP_HOT,
    }
    new_im_paths = {}
    heat_im_paths = {}
    heat_ims = {}

    # read the image bounding box again
    # im1 = mr_image.read_region(
    #     (startX, startY), wsi_mag_level, (extentX_wsi, extentY_wsi))
    im1 = read_region_by_power(
        mr_image, (startX, startY), wsi_mag_power, (extentX_wsi, extentY_wsi))

    im1 = np.array(im1)[:, :, :3]
    print(score_map.shape)
    print(im1.shape)

    for color_name, color_map in color_maps.items():

        heatmap = cv2.applyColorMap(
            np.uint8(255 * score_map), color_map)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blender = HeatmapBlender(im1, heatmap, im_mask)
        alpha_mask = np.where(im_mask > 0, alpha, 0).astype(np.float32)
        alpha_mask = np.expand_dims(alpha_mask, [2])

        new_im = heatmap * alpha_mask
        alpha_mask = 1-alpha_mask
        new_im = new_im + im1 * alpha_mask
        # fin = cv2.addWeighted(np.array(im1)[:, :, 0:3], 0.8, heatmap, 0.2, 1)

        # blender = HeatmapBlender(im1, heatmap, im_mask)
        # fin = blender.get_final_image(
        #     ksize=(151, 151), sigmaX=None, tolerance=10, save=False)

        new_im = Image.fromarray(new_im.astype(np.uint8))
        heat_im = Image.fromarray(heatmap)
        new_im_path = os.path.join(
            hm_save_folder, f'{save_key}_{color_name}.png')
        heat_im_path = os.path.join(
            hm_save_folder, f'{save_key}_{color_name}_mask.png')
        new_im.save(new_im_path, optimize=True)
        heat_im.save(heat_im_path, optimize=True)
        new_im_paths[color_name] = new_im_path
        heat_im_paths[color_name] = heat_im_path
        heat_ims[color_name] = heat_im
        print(f"{save_key} heat map saved ({color_name})")

    results = {
        'crop_path': crop_path,
        'new_im_paths': new_im_paths,
        'heat_im_paths': heat_im_paths,
    }
    return results
