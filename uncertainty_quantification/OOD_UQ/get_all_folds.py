import glob
from os.path import join, basename, dirname
import os
import shutil
# sstr = 'hydra_logs_CV/wBenign/fold*/train/runs/*_best/data/epoch_*/postnet_*.csv'
sstr = ['hydra_logs_CV/wBenign/fold*/eval/runs/*/data/epoch_*/postnet_*.csv' , 
    'hydra_logs_CV/wBenign/fold*/eval/runs/*/data/epoch_*/postnet_*.npy']
outfolder = 'summary_CTrans_wBenign_classWeighted_rerun'
os.makedirs(outfolder, exist_ok=True)
csvs = glob.glob(sstr[0]) + glob.glob(sstr[1])
# copy all csvs to a folder
for csv in csvs:
    shutil.copy(csv, outfolder)