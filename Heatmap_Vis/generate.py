Import Create_Heatmap from heatmap_utils
Import create_heatmap from heatmap_utils

def main(args):
    print('processing gets started! ')
    CLASS_NUMB = args.outdim
    H5_SAVE_FOLDER = args.h5_folder
    HEAT_SAVE_FOLDER = args.heatmap_folder
    SLIDE_PATH = args.slide_path
    PATCH_SAVE_FOLDER = args.patch_folder
    H5_EXIST_PATH = args.h5_file
    MODEL_PATH = args.model_path
    COLUMN_PROB = args.column
    PICKLE_PATH = args.pickle
    DEVICE = args.device
    NORM = args.norm_heat
    ONLY_H5 = args.onlyh5

    if int(args.region[0]) == int(args.region[1]) == int(args.region[2]) == int(args.region[3]):
        mr_image = openslide.OpenSlide(SLIDE_PATH)
        orig_xDim = mr_image.level_dimensions[0][0]
        orig_yDim = mr_image.level_dimensions[0][1]
        bl_X, bl_Y, tr_X, tr_Y = 0,0, orig_xDim, orig_yDim
    else:
        bl_X, bl_Y, tr_X, tr_Y = int(args.region[0]),int(args.region[1]),int(args.region[2]),int(args.region[3])

    heatmap_mask = Create_Heatmap(SLIDE_PATH)
    row = {'labels': args.label,'bottom_left_X': bl_X, 'bottom_left_Y': bl_Y , 'top_right_X': tr_X, 'top_right_Y':tr_Y}
    label, bl_X, bl_Y = row['labels'], row['bottom_left_X'], row['bottom_left_Y']
    h5_file_path = H5_SAVE_FOLDER+SLIDE_PATH.split("/")[-1]+f'+NA+{bl_X}+{bl_Y}+{tr_X}+{tr_Y}.hdf5'
    print(h5_file_path)
    import os.path
    from os import path

    if not path.exists(h5_file_path) and H5_EXIST_PATH == None:
        heatmap_mask.tile_patch(PATCH_SAVE_FOLDER+SLIDE_PATH.split("/")[-1],row,HEAT_SAVE_FOLDER)
        save_h5(PATCH_SAVE_FOLDER+SLIDE_PATH.split("/")[-1], h5_file_path,clean=False)
        if not ONLY_H5:
            print('start rendering heats')
            create_heatmap(h5_file_path,MODEL_PATH,HEAT_SAVE_FOLDER,heatmap_mask.metric,row,SLIDE_PATH,col =COLUMN_PROB,no_of_classes=CLASS_NUMB,csv=PICKLE_PATH,device_id=DEVICE,normalize=NORM)
    else:
        print('{} h5 existed'.format(h5_file_path))
        create_heatmap(h5_file_path,MODEL_PATH,HEAT_SAVE_FOLDER,heatmap_mask.metric,row,SLIDE_PATH,col =COLUMN_PROB,no_of_classes=CLASS_NUMB,csv=PICKLE_PATH,device_id=DEVICE,normalize=NORM)



    print('Visualization Done!')



import argparse
parser = argparse.ArgumentParser(description='Configurations')

parser.add_argument('--slide-path',
                    help='directory for one BWH WSI',
                    default='', type=str)

parser.add_argument('--h5-folder',
                    help='folder to save hdf5 data',
                    default='', type=str)


parser.add_argument('--heatmap-folder',
                    help='folder to save all heatmaps',
                    default='', type=str)

parser.add_argument('--pickle',
                    help='pickle dataframe for tf predictions',
                    default=None, type=str)

parser.add_argument('--device',
                    help='which gpu',
                    default='1', type=str)

parser.add_argument('--onlyh5', action='store_true')

parser.add_argument('--label',
                    help='label for selected region',
                    default='NA', type=str)

parser.add_argument('--h5-file',
                    help='path for using a h5 file',
                    default=None, type=str)


parser.add_argument('--column',
                    help='based on which prob column',
                    default='Low', type=str)

parser.add_argument('--outdim', type=int, help='how many classes to predict', default=2)

parser.add_argument('--patch-folder',
                    help='folder to save all ptaches',
                    default='', type=str)

parser.add_argument('--norm-heat', action='store_true', default=False, help='normalize the prediction weights for rendering heatmap')


parser.add_argument('--model-path',
                    help='path for a trained model',
                    default='', type=str)

parser.add_argument('--region','--list', nargs='+', help='a selected region to visualize', required=True)

args = parser.parse_args()

if __name__ == '__main__':

    print("started!")
    import datetime

    now = datetime.datetime.now()
    print("Starting date and time : ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    print("Starting Execution Time:", now)
    results = main(args)
    print("finished!")
    print("end script")
