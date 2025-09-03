## WSI_tile_extraction.py

import matplotlib.pyplot as plt
import os
import openslide
import numpy as np
import math
from jsonc_parser.parser import JsoncParser
from PIL import Image
from tqdm import tqdm
from skimage.feature import blob_log, blob_dog, blob_doh
from scipy import ndimage
from skimage.color import rgb2gray
from argparse import ArgumentParser
import pandas as pd


parser = ArgumentParser()
# Required arguments
parser.add_argument(
    "--infile",
    help="""Input WSI File """,
    type=str,
    default= "" ,
)

parser.add_argument(
    "--params",
    help="""Param File """,
    type=str,
    default="image_patching_script/tile_params_default500_small.jsonc",
)

parser.add_argument(
    "--outpath",
    help="""Output Path """,
    type=str,
    default="",
)
parser.add_argument(
    "--proj", help="""Project name (folder prefix) """, type=str, default="Debug"
)
parser.add_argument(
    "--blob_fun", help="""Blob function  """, type=str, default="log",choices=["log",'dog','doh']
)
parser.add_argument(
    "--overwrite_if_exist",
    help="""Overwrite file is already existed """,
    type=bool,
    default=True,
)
parser.add_argument(
    "--output_tiles",
    help="""Output tiles """,
    type=bool,
    default=True,
)
parser.add_argument(
    "--debug", help="""Debug using small dataset""", type=bool, default=False
)
args = parser.parse_args()


TQDM_INVERVAL = 10
###### read params from JSON file
BLOB_MAG_LEVEL = 5
locals().update(JsoncParser.parse_file(args.params))
hueLowerBound = hueLowerBound / 360 * 255
hueUpperBound = hueUpperBound / 360 * 255
saturationLowerBound = saturationLowerBound * 255
#
params = JsoncParser.parse_file(args.params)
print("=======  Parameters:  =======")
for key in params.keys():
    print(f"  {key}:\t{params[key]}")
###########################
print("=======      End      =======")

if args.debug:
    xStride = 250
    yStride = 250
    TQDM_INVERVAL = 1
    args.overwrite_if_exist=True
    args.infile = "" 


filename = os.path.basename(args.infile)

outputDirBase = os.path.join(
    args.outpath,
    f"{args.proj}_{xPatch}dense_max{numPatches}_Q{top_Q}_Zoom{IMG_MAG_LEVEL}X/",
)

if not os.path.isdir(outputDirBase):
    try:
        os.makedirs(outputDirBase)
    except:
        pass
outputDir = os.path.join(outputDirBase, filename)
outputDirStats = os.path.join(outputDirBase, "zStats/")
outputDirThumbNail = os.path.join(outputDirBase, "thumbnail/")


def rgbHueSat(pix):
    # pixImg=Image.fromarray(pix)

    pixHsv = pix.convert("HSV")
    pixHsvArray = np.array(pixHsv, dtype=np.float32)
    return pixHsvArray[:, :, 0], pixHsvArray[:, :, 1]


def OpticalDensityThreshold(I, Io=240, beta=0.15):
    # calculate optical density
    OD = -np.log((I.astype(np.float32) + 1) / Io)

    # remove transparent pixels
    ODhat = ~np.any(OD < beta, axis=2)

    return OD, ODhat


def get_subsample_rate(slide, mag_power=20):
    ## Get the subsample rate compared to level 0
    ## (necessary since levels in tcgaGBM data is not downsampled by the power of 2)
    mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    assert (
        mag >= mag_power
    ), f"Magnification of the slide ({mag}X) is smaller than the desired magnification ({mag_power}X)."

    ds_rate = mag / mag_power
    return int(ds_rate)


def get_sampling_params(slide, mag_power=20):
    # Get the optimal openslide level and subsample rate, given a magnification power
    mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    assert (
        mag >= mag_power
    ), f"Magnification of the slide ({mag}X) is smaller than the desired magnification ({mag_power}X)."
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
    read_width = tuple([int(opt_ds_rate * x) for x in width])
    im1 = slide.read_region(start, opt_level, read_width)
    if opt_ds_rate != 1:
        im1 = im1.resize(width, resample=Image.LINEAR)
    return im1


def main():
    # for filename in os.listdir(dirname):
    print(filename)
    mr_image = openslide.OpenSlide(args.infile)
    # level = get_zoom_level(mr_image, mag_power=IMG_MAG_LEVEL)
    # search_level = get_zoom_level(mr_image, mag_power=SEARCH_MAG_LEVEL)
    ds_rate = get_subsample_rate(mr_image, mag_power=IMG_MAG_LEVEL)
    search_ds_rate = get_subsample_rate(mr_image, mag_power=SEARCH_MAG_LEVEL)
    blob_ds_rate = get_subsample_rate(mr_image, mag_power=BLOB_MAG_LEVEL)

    if os.path.isdir(outputDir):
        if not args.overwrite_if_exist:
            raise FileExistsError(
                f"Directory {outputDir} already exists. Exited for safety reason (consider saving in another root directory or setting --overwrite_if_exist True )"
            )
        else:
            print(
                f"Directory {outputDir} already exists. Will overwrite the existing file."
            )
        # shutil.rmtree(outputDir)    os.mkdir(outputDir)
    os.makedirs(outputDirStats, exist_ok=True)
    os.makedirs(outputDirThumbNail, exist_ok=True)

    lvl_xStride = xStride * ds_rate
    lvl_yStride = yStride * ds_rate
    # ds = mr_image.getLevelDownsample(level)
    xDim = mr_image.level_dimensions[0][0]
    yDim = mr_image.level_dimensions[0][1]

    AllValidTally = []
    nonAllCriteriaSumCounts = []
    blobsCount = []

    #
    nonEmpty = 0
    AllValidTallyFileName = os.path.join(
        outputDirStats, f"AllValidSatTally{filename}.csv"
    )

    # if args.debug:
    ## Save thumbnail
    max_level = mr_image.level_count - 1
    # thumbnail = mr_image.read_region((0, 0), max_level,mr_image.level_dimensions[max_level])
    thumbnail_size = 1000
    thumbnail = mr_image.get_thumbnail((thumbnail_size, thumbnail_size))
    thumbnail_name = os.path.join(outputDirThumbNail, f"{filename}.jpg")
    thumbnail.convert("RGB").save(thumbnail_name)
    if args.debug:  # and np.sqrt(np.mean(OD_LoG**2)) > LOGLowerBound:
        thumbnail.convert("RGB").save("thumbnail.jpg")

    pbar = tqdm(range(int(xDim / lvl_xStride)), mininterval=TQDM_INVERVAL)
    count = 0
    sub_size = (
        int(xPatch / search_ds_rate * ds_rate),
        int(yPatch / search_ds_rate * ds_rate),
    )

    blob_sub_size = (
        int(xPatch / blob_ds_rate * ds_rate),
        int(yPatch / blob_ds_rate * ds_rate),
    )


    count_threshold = area_threhold * sub_size[0] * sub_size[1]
    nonBlack_count_threshold = (1 - BlackAreaThreshold) * sub_size[0] * sub_size[1]

    AllValidTallyList = []
    for i in pbar:
        # print(f"{i}/{int(xDim/lvl_xStride)}")
        pbar.set_description(f" {len(AllValidTally)} Valid, {count} Total")

        for j in range(int(yDim / lvl_yStride)):
            count = count + 1
            # print(i,j)
            try:
                image_patch = read_region_by_power(
                    mr_image,
                    (lvl_xStride * i, lvl_yStride * j),
                    SEARCH_MAG_LEVEL,
                    sub_size,
                )

                pix = np.array(image_patch)
                pix = pix[:, :, 0:3]
                # pixR = pix[:, :, 0]
                # pixG = pix[:, :, 1]
                # pixB = pix[:, :, 2]
                # pixRange = np.ptp(pix, axis=2)
                # nonEmpty = (pixRange > nonEmptyRangeThreshold)
                
                # OD
                OD, nonEmpty = OpticalDensityThreshold(pix, Io=OD_Io, beta=OD_beta)
                nonEmpty = np.where(nonEmpty, 1, 0)
                nonEmptySum = nonEmpty.sum()
                if nonEmptySum < count_threshold:
                    continue

                ## log
                OD_gray = rgb2gray(OD)
                if LOGLowerBound > 0:
                    # to save time, only run LoG filter is LOGLowerBound > 0
                    OD_LoG = ndimage.gaussian_laplace(OD_gray, sigma=LOGsigma)
                    OD_abs_LoG = np.abs(OD_LoG)
                else:
                    OD_abs_LoG = np.ones_like(OD_gray)
                nonLoG = np.where(OD_abs_LoG > LOGLowerBound, 1, 0)
                nonLoGSum = np.sum(nonLoG)
                if nonLoGSum < count_threshold:
                    continue
                ## Sat and hue
                pixHue, pixSat = rgbHueSat(image_patch)
                nonHue = np.where(np.logical_or(pixHue < hueLowerBound, pixHue > hueUpperBound), 1, 0)
                nonSat = np.where(pixSat > saturationLowerBound, 1, 0)
                nonHueSum = np.sum(nonHue)
                if nonHueSum < count_threshold:
                    continue
                nonSatSum = np.sum(nonSat)
                if nonSatSum < count_threshold:
                    continue
                ## Black
                nonBlack = np.where(np.min(pix, axis=2) > BlackThresold, 1, 0)
                nonBlackSum = np.sum(nonBlack)
                if nonBlackSum < nonBlack_count_threshold:
                    continue
                ##
                # nonLoG = np.where(OD_LoG < -LOGLowerBound,1,0)
                nonAllCriteria = nonEmpty * nonHue * nonSat * nonBlack * nonLoG

                nonAllCriteriaSum = np.sum(nonAllCriteria)
                # nonAllCriteriaSum = np.min(nonAllCriteriaSum,nonBlackSum)

                # nonAllCriteria =  np.logical_and(np.logical_and(nonEmpty, nonHue),nonSat)

                if args.debug:  # and np.sqrt(np.mean(OD_LoG**2)) > LOGLowerBound:
                    image_patch.convert("RGB").save("test_patch.jpg")

                if (
                    nonAllCriteriaSum > count_threshold
                    and nonBlackSum > nonBlack_count_threshold
                ):
                    ## read higher-res image for blob detection
                    if BLOB_MAG_LEVEL != SEARCH_MAG_LEVEL:
                        
                        image_patch = read_region_by_power(
                            mr_image,
                            (lvl_xStride * i, lvl_yStride * j),
                            BLOB_MAG_LEVEL,
                            blob_sub_size,
                        )
                        pix = np.array(image_patch)
                        pix = pix[:, :, 0:3]
                        OD, _ = OpticalDensityThreshold(pix, Io=OD_Io, beta=OD_beta)
                        OD_gray = rgb2gray(OD)

                    if args.blob_fun == 'log':
                        blobs = blob_log(
                            OD_gray,
                            min_sigma=blobsRadiusLowerBound,
                            max_sigma=blobsRadiusUpperBound,
                            num_sigma=blobNumSigma,
                            threshold=blobThreshold,
                        )
                    elif args.blob_fun == 'dog':
                        blobs = blob_dog(
                            OD_gray,
                            min_sigma=blobsRadiusLowerBound,
                            max_sigma=blobsRadiusUpperBound,
                            threshold=blobThreshold,
                        )                   
                    elif args.blob_fun == 'doh':
                        blobs = blob_doh(
                            OD_gray,
                            min_sigma=blobsRadiusLowerBound,
                            max_sigma=blobsRadiusUpperBound,
                            num_sigma=blobNumSigma,
                            threshold=blobThreshold,
                        ) 
                    # blobs = BLOB_FUN(
                    #     OD_gray,
                    #     min_sigma=blobsRadiusLowerBound,
                    #     max_sigma=blobsRadiusUpperBound,
                    #     num_sigma=blobNumSigma,
                    #     threshold=blobThreshold,
                    # )
                    # blobs = blob_dog(OD_gray, max_sigma=blobsRadiusUpperBound, threshold=.3)
                    blobs = blobs[
                        np.logical_and(
                            blobs[:, 2] < blobsRadiusUpperBound,
                            blobs[:, 2] > blobsRadiusLowerBound,
                        ),
                        :,
                    ]
                    if BLOB_ONLY_IN_VALID:
                        idx_valid = np.argwhere(
                            np.array(
                                [
                                    nonAllCriteria[int(x), int(y)]
                                    for x, y in zip(blobs[:, 0], blobs[:, 1])
                                ]
                            )
                        ).flatten()
                        blobs = blobs[idx_valid, :]

                    if blobs.shape[0] > blobsNumLowerBound:
                        if (
                            args.debug
                        ):  # and np.sqrt(np.mean(OD_LoG**2)) > LOGLowerBound:
                            # image_patch.convert('RGB').save('test_patch.jpg')

                            # plt.close()
                            # ax = plt.figure(figsize=(8, 8), dpi=300)
                            # plt.imshow(pix,vmax=255)

                            for blob in blobs:
                                y, x, r = blob
                                c = plt.Circle(
                                    (x, y), r, color="red", linewidth=1, fill=False
                                )
                                plt.gca().add_patch(c)
                            plt.gca().set_axis_off()
                            plt.savefig("test_blob.jpg")

                            plt.close()
                            ax = plt.figure(figsize=(8, 8), dpi=300)

                            plt.subplot(331)
                            plt.imshow(pix, vmax=255)

                            for blob in blobs:
                                y, x, r = blob
                                c = plt.Circle(
                                    (x, y), r, color="red", linewidth=1, fill=False
                                )
                                plt.gca().add_patch(c)
                            plt.gca().set_axis_off()

                            plt.title(f"orig, {blobs.shape[0]} blobs")

                            plt.subplot(332)
                            plt.imshow(OD, vmin=0.15)
                            plt.gca().set_axis_off()
                            plt.title("OD")

                            plt.subplot(333)
                            plt.imshow(nonEmpty, vmin=0, vmax=1)
                            plt.gca().set_axis_off()
                            plt.title("nonEmpty")

                            plt.subplot(334)
                            plt.imshow(OD_abs_LoG, vmin=LOGLowerBound)
                            plt.gca().set_axis_off()
                            # plt.imshow(-OD_LoG,vmin=LOGLowerBound)
                            # plt.colorbar()
                            plt.title("OD_abs_LoG")

                            plt.subplot(335)
                            plt.imshow(nonHue, vmin=0, vmax=1)
                            plt.gca().set_axis_off()
                            plt.title("nonHue")

                            plt.subplot(336)
                            # plt.imshow(pixSat,vmin=saturationLowerBound)
                            # plt.title('pixSat')
                            plt.imshow(nonSat, vmin=0, vmax=1)
                            plt.gca().set_axis_off()
                            plt.title("nonSat")

                            plt.subplot(337)
                            plt.imshow(nonBlack, vmin=0, vmax=1)
                            plt.gca().set_axis_off()
                            plt.title("nonBlack")

                            plt.subplot(338)
                            plt.imshow(nonAllCriteria, vmin=0, vmax=1)
                            plt.gca().set_axis_off()
                            plt.title("nonAllCriteria")

                            plt.subplot(339)
                            plt.hist(OD.flatten(), bins=100)
                            plt.gca().set_axis_off()
                            plt.title("nonAllCriteria")
                            plt.savefig("test.jpg")

                            a = 0

                        # AllValidTallyFile.write(str(i*lvl_xStride) + "\t" + str(j*lvl_yStride) + "\t" + str(nonEmptySum) + "\t" + str(nonHueSum) + "\t" + str(nonAllCriteriaSum) + "\n")
                        AllValidTallyList.append(
                            {
                                "X": i * lvl_xStride,
                                "Y": j * lvl_yStride,
                                "BlobCount": blobs.shape[0],
                                "nonAllCriteriaSum": nonAllCriteriaSum,
                                "nonHueSum": nonHueSum,
                                "nonSatSum": nonSatSum,
                                "nonLoGSum": nonLoGSum,
                                "nonBlackSum": nonBlackSum,
                            }
                        )
                        # nonEmptyTallyFile.write(str(i*lvl_xStride) + "\t" + str(j*lvl_yStride) + "\t" + str(nonEmptySum) + "\n")
                        # nonEmptyTally.append([int(i), int(j), int(nonEmptySum)])
                        AllValidTally.append([int(i), int(j), int(nonEmptySum)])
                        nonAllCriteriaSumCounts.append(int(nonAllCriteriaSum))
                        blobsCount.append(blobs.shape[0])

            except:
                pix = 0
    # valid_tiles = np.argwhere(np.array(nonAllCriteriaSumCounts)> count_threshold )
    # nonEmptyTallyFile.close()
    df_AllValid = pd.DataFrame.from_records(AllValidTallyList)
    # AllValidTallyFile.close()
    total_tiles = int(xDim / lvl_xStride) * int(yDim / lvl_yStride)
    AllValidTally = np.stack(AllValidTally, axis=0)
    nonAllCriteriaSumCounts = np.array(nonAllCriteriaSumCounts)
    blobsCount = np.array(blobsCount)

    print(f"Total Tiles: {total_tiles}")
    print(f"Nonempty Tiles: {len(AllValidTally)}")

    SortVal = (
        nonAllCriteriaSumCounts
        / np.std(nonAllCriteriaSumCounts)
        * blobsCount
        / np.std(blobsCount)
    )
    idx_sort = np.flip(np.argsort(SortVal))

    df_AllValid = df_AllValid.loc[idx_sort]
    df_AllValid["rank"] = np.arange(1, df_AllValid.shape[0] + 1)
    df_AllValid.to_csv(AllValidTallyFileName)
    AllValidTally = AllValidTally[idx_sort, :]

    nSelected = min(int(numPatches), int(len(AllValidTally) * top_Q))

    # nStart=0
    if args.output_tiles:
        for i in tqdm(range(nSelected), mininterval=TQDM_INVERVAL):
            # for i in tqdm(range(len(AllValidTally)),mininterval=TQDM_INVERVAL):
            # print(i)
            outputFileName = os.path.join(
                outputDir,
                f"tile{i}_"
                + filename
                + "_"
                + str(lvl_xStride * AllValidTally[i, 0])
                + "_"
                + str(lvl_yStride * AllValidTally[i, 1])
                + ".jpg",
            )

            patch_size = (xPatch, yPatch)

            image_patch = read_region_by_power(
                mr_image,
                (lvl_xStride * AllValidTally[i, 0], lvl_yStride * AllValidTally[i, 1]),
                IMG_MAG_LEVEL,
                patch_size,
            )

            # image_patch = mr_image.read_region((lvl_xStride * AllValidTally[i,0], lvl_yStride * AllValidTally[i,1]), level, (xPatch, yPatch))
            pix = np.array(image_patch)
            pix = pix[:, :, 0:3]
            # scipy.misc.toimage(pix, cmin=0, cmax=255).save(outputFileName)
            Image.fromarray(pix).save(outputFileName)
            # returnValue = os.system("cp " + selectedImageFileName + " " + outputDir)

    print("done")


if __name__ == "__main__":
    main()
