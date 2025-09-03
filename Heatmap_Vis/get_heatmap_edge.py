
import argparse
from blender import *
from typing import Literal

import pandas as pd
from heatmap_utils import *

import sys
import os
from datetime import datetime
import cv2
import numpy as np
import PIL
from PIL import Image, ImageFilter
from glob import glob
from model_loaders import *
from tqdm import tqdm
from heatmap_utils import get_mask



MASK_HUE = 120
BG_VAL = 220
THRESHOLD_HIGH = 100
THRESHOLD_LOW = 20
OUTPATH = 'WSI_heatmap/Edge_HighAndLow'
OUTPATH_OVERLAY = 'WSI_heatmap/Edge_HighAndLow_overlay'
os.makedirs(OUTPATH,exist_ok=True)
os.makedirs(OUTPATH_OVERLAY,exist_ok=True)




parser = argparse.ArgumentParser(description='Configurations')

parser.add_argument('--start', type=int,
                    default=0)
parser.add_argument('--end', type=int,
                    default=10000)
args = parser.parse_args()


def fill_mask(img,mask,fill_val=[255,0,0]):
    img = img.astype(np.float32)
    img2 =  np.zeros_like(img)
    for i in range(3):
        img2[:,:,i] = mask * fill_val[i]
    mask = np.tile(np.expand_dims(mask,axis=2),[1,1,3])
    img_new = mask*img2 + (1-mask) * img
    return img_new
def get_edge(slide_file,outname=None):
    heatmap_file = slide_file.replace('_crop','_JET_mask')
    overlay_file = slide_file.replace('_crop','_JET')
    if outname is not None:
        out_file_overlay = os.path.join(OUTPATH_OVERLAY,outname)
        out_file = os.path.join(OUTPATH,outname)
    else:
        out_file  = slide_file.replace('_crop','_JET_edge')
        out_file_overlay = os.path.join(OUTPATH_OVERLAY,os.path.basename(out_file))
        out_file = os.path.join(OUTPATH,os.path.basename(out_file))
    if os.path.isfile(out_file):
        # pass
        return

    slide_img = cv2.imread(slide_file)
    overlay_img = cv2.imread(overlay_file)
    heatmap_img = cv2.imread(heatmap_file)
    heatmap_hue = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2HSV)[:,:,0]
    valid_mask = get_mask(Image.fromarray(slide_img))
    mask_h = np.where(np.logical_and(heatmap_hue>THRESHOLD_HIGH,valid_mask>0),1.0,0.0)
    mask_l = np.where(heatmap_hue<THRESHOLD_LOW,1.0,0.0)
    ih = Image.fromarray(mask_h)
    # Detecting Edges on the Image using the argument ImageFilter.
    ih = ih.convert("L")
    ih = ih.filter(ImageFilter.FIND_EDGES)
    ih = np.array(ih)
    ih = cv2.blur(ih,(5,5))
    mask_h = np.where(ih>0,1,0)


    il = Image.fromarray(mask_l)
    il = il.convert("L")
    il = il.filter(ImageFilter.FIND_EDGES)
    il = np.array(il)
    il = cv2.blur(il,(5,5))
    mask_l = np.where(il>0,1,0)

    filled = fill_mask(slide_img,mask_l,fill_val=[0,0,160])
    filled = fill_mask(filled,mask_h,fill_val=[256,0,0])
    cv2.imwrite(out_file,filled)
    
    filled = fill_mask(overlay_img,mask_l,fill_val=[0,0,160])
    filled = fill_mask(filled,mask_h,fill_val=[256,0,0])
    cv2.imwrite(out_file_overlay,filled)


if __name__ == "__main__":
    df_selected = pd.read_csv('Heatmap-List.csv')
    

    img_list = glob.glob(
        os.path.join('WSI_heatmap','Heatmaps_Results_*','Heats_Mag2.5X','*_crop.png')
    )
    img_list_selected = []
    df_selected['file'] = ''
    
    for i in range(df_selected.shape[0]):
        ID = df_selected['Slide ID'].loc[i]
        contain_ID = np.array([ ID in x for x in img_list])
        idx = np.argwhere(contain_ID).flatten()[0]
        df_selected['file'].loc[i] = img_list[idx]

   
    start = args.start
    end = np.minimum(args.end,df_selected.shape[0])
    df_selected = df_selected.loc[start:end-1].reset_index()
    for idx in tqdm(range(df_selected.shape[0])):
        row = df_selected.loc[idx]
        img = row['file']
        outname = row['Code'] + '-HeatsContour.png'
        get_edge(img,outname)
