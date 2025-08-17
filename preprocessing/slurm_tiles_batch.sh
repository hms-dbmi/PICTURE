#!/bin/bash
# ==================================================================
# Usage:
#   sbatch slurm_tiles.sh <slide_folder> <patch_folder> [<n_parts> <part>]
#
# Notes:
#   <n_parts> <part> is optional. If you leave it out, it will be 0 and 1.
#   Adjust memory according to your keep_random_n and magnifications
#   arguments.
# ==================================================================
#SBATCH -c 1
#SBATCH -t 1:00:00
#SBATCH -p short
#SBATCH --account=yu_ky98
#SBATCH --mem=1G
#SBATCH -o logs/tile_extraction_%A_%a.log
#SBATCH -e logs/tile_extraction_%A_%a.log
#SBATCH --array=0-100    # <- change this to the number of jobs you want to split the data into

## === PARAMETERS FOR THE TILE EXTRACTION ===
n_parts=100               # <- change this to the number of jobs you want to split the data into
slide_folder="/n/data2/hms/dbmi/kyu/lab/shl968/WSI_for_debug" # <- change this to the path containing your WSIs (subfolders/nested subfolders are okay)
patch_folder="/n/scratch/users/s/shl968/WSI_prep_test"        # <- change this to the path to save your patches
# NOT RECOMMEND USING 40X MAGNIFICATION IF YOU ARE NOT SURE IF ALL YOUR SLIDES ARE 40X
MAGNIFICATION=20       
## ==============================================




# module restore default
# source activate HTAN_env
# python3 --version
module load gcc/6.2.0 cuda/10.2 python/3.6.0 libpng/1.6.26 jpeg/9b tiff/4.0.7 glib/2.50.2 libxml/2.9.4 freetype/2.7 libffi/3.2.1 fontconfig/2.12.1 harfbuzz/1.3.4 cairo/1.14.6 openjpeg/2.2.0 openslide/3.4.1
source activate /n/data2/hms/dbmi/kyu/lab/NCKU/conda_env/moe


IDX=$((SLURM_ARRAY_TASK_ID))
part=$IDX

python create_tiles.py \
    --slide_folder $slide_folder \
    --patch_folder $patch_folder \
    --patch_size 224 \
    --stride 224 \
    --output_size 224 \
    --tissue_threshold 80 \
    --magnifications $MAGNIFICATION \
    --n_workers 1 \
    --n_parts $n_parts \
    --part $part \
    --only_coords
