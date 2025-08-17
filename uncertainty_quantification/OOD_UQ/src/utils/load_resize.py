from multiprocessing import Pool 
import glob
from PIL import Image
import os
import time
import numpy as np
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None
from tqdm.contrib.concurrent import process_map
path_list = [os.path.join(dirpath,filename) for dirpath, _, filenames in os.walk('/n/data2/hms/dbmi/kyu/lab/datasets/IvyGap/HE/') for filename in filenames if filename.endswith('.jpg')]
def resize(path):
    path_compressed = path.replace('HE', 'compressed')
    path_compressed = path_compressed.replace('jpg','png')    
    
    # create directory if it doesn't exist
    dir_compressed = '/'.join(path_compressed.split('/')[:-1])
    if not os.path.isdir(dir_compressed):
        os.makedirs(dir_compressed)
    im = Image.open(path)
    
    is_segmentation_map = path[-5] == 'A'
    im = im.resize((224*4,224*4), resample=Image.Resampling.NEAREST if is_segmentation_map else Image.Resampling.BICUBIC)
    
    
    if is_segmentation_map:
        convert(im, path_compressed)
    else:
        path_compressed = path_compressed.replace('jpg','png')
        im = im.save(path_compressed)
    
def convert(image, path_compressed):
    # Step 1: Define palette
    palette = np.array([[33, 143, 166], [210,5,208], [5, 208, 4], [37, 209, 247], [6, 208,170], [255, 102, 0], [5,5,5], [255, 255,255]])
    #mapping = {(33, 143, 166): 0, (210,5,208): 1, (5, 208, 4): 3, (37, 209, 247): 8, (6, 208,170): 8, (255, 102, 0): 6, (5,5,5): 8, (255, 255,255): 9}
    mapping = {(33, 143, 166): 0, (210,5,208): 1, (5, 208, 4): 3, (37, 209, 247): 5, (6, 208,170): 4, (255, 102, 0): 6, (5,5,5): 8, (255, 255,255): 9}
    reducing = {0:0, 1:1, 3:2, 5:3, 4:3, 5:3, 6:3, 8:3, 9:4}
   
    # Step 2: Create/Load precalculated color cube
    try:
        # for all colors (256*256*256) assign color from palette
        precalculated = np.load('data/view.npz')['color_cube']
    except:
        precalculated = np.zeros(shape=[256,256,256,3])
        for i in tqdm(range(256)):
            for j in range(256):
                for k in range(256):
                    index = np.argmin(np.sqrt(np.sum(((palette)-np.array([i,j,k]))**2,axis=1)))
                    precalculated[i,j,k] = reducing[mapping[tuple(palette[index])]]
        np.savez_compressed('data/view', color_cube = precalculated)
            
    def get_view(color_cube,image):
        image = np.array(image)
        shape = image.shape[0:2]
        indices = image.reshape(-1,3)
        # pass image colors and retrieve corresponding palette color
        new_image = color_cube[indices[:,0],indices[:,1],indices[:,2]]
    
        return new_image.reshape(shape[0],shape[1],3).astype(np.uint8)

    image_converted = get_view(precalculated, image)
        
    Image.fromarray(image_converted).save(path_compressed)

if __name__ == '__main__':
    images = glob.glob('/n/data2/hms/dbmi/kyu/lab/datasets/IvyGap/HE/268415286_273294269/*.jpg')
    images = []
    for root, dirs, files in os.walk('/n/data2/hms/dbmi/kyu/lab/datasets/IvyGap/HE/'):
        if files:
            images.append(sorted(files)[:2])
            
            
    images = [item for sublist in images for item in sublist]
    images = ['/n/data2/hms/dbmi/kyu/lab/datasets/IvyGap/HE/'+'_'.join(name.split('_')[:2])+'/'+name for name in images ]

    for image in tqdm(images):
        resize(image)

