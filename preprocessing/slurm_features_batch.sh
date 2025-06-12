#!/bin/bash
# ==================================================================
# Usage:
#   sbatch slurm_features.sh <path_to_patches> <path_to_features>
#
# Notes:
#   I experimented with batch sizes, and 2048 to 3072 seem to work.
#   You need to first create a conda environment and a huggingface
#   token (for UNI) to run this script. Add the token as hf_token
#   argument.
# ==================================================================

#!/bin/bash
#SBATCH -c 2                         # Request four cores
#SBATCH -t 1:00:00                    # Runtime in D-HH:MM format
#SBATCH -p gpu_quad                     # Partition to run in
##SBATCH --account=yu_ky98_contrib     #
#SBATCH --gres=gpu:1                   # Number of GPUS
#SBATCH --mem=6G                     # Memory total in MiB (for all cores)
#SBATCH -o ./logs/%j_%N_%x.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./logs/%j_%N_%x.out                 # File to which STDERR will be written, including job ID (%j)

#SBATCH -o logs/feature_extraction_%A_%a.log
#SBATCH -e logs/feature_extraction_%A_%a.log
#SBATCH --array=0    # <- change this to the number of jobs you want to split the data into


# module restore moe   # O2 modules. Only gcc & cuda is strictly necessary
module purge
module load gcc/6.2.0 cuda/10.2 python/3.6.0
source activate /n/data2/hms/dbmi/kyu/lab/NCKU/conda_env/moe
## === PARAMETERS FOR THE FEATURE EXTRACTION ===
## ===         (CHANGE THIS IF NEEDED)       ===
n_parts=1 # <- change this to the number of jobs you want to split the data into
slide_folder="/n/data2/hms/dbmi/kyu/lab/shl968/WSI_for_debug"   # <- change this to the path to your WSI
patch_folder="/n/scratch/users/s/shl968/WSI_prep_test"          # <- change this to the path to your patch coords (from create_tiles.py)
feat_folder="/n/scratch/users/s/shl968/WSI_feat_test"           # <- change this to the path to save your features
STAIN_NORM=true        # <- change this to true if you want to stain normalize your patches
TARGET_MAG=20           # <- change this to the target magnification for your features
## ==============================================

## == LOAD MODULES ==
# module restore moe   # O2 modules. Only gcc & cuda is strictly necessary
# source activate moe  # install the moe environment using requirements.txt, plus the OpenSlide as described in the README
module load gcc/6.2.0 cuda/10.2 python/3.6.0 libpng/1.6.26 jpeg/9b tiff/4.0.7 glib/2.50.2 libxml/2.9.4 freetype/2.7 libffi/3.2.1 fontconfig/2.12.1 harfbuzz/1.3.4 cairo/1.14.6 openjpeg/2.2.0 openslide/3.4.1
## ==================
source activate /n/data2/hms/dbmi/kyu/lab/NCKU/conda_env/moe



IDX=$((SLURM_ARRAY_TASK_ID))
part=$IDX

export CHIEF_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/chief.pth"
export CTRANS_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/ctranspath.pth"

# python /home/bal753/sys_info.py

echo 
nvidia-smi
echo


model=ctrans,chief 
python create_features.py \
 --wsi_folder $slide_folder \
 --patch_folder $patch_folder \
 --feat_folder $feat_folder \
 --n_parts $n_parts \
 --part $part \
 --models $model \
 --target_mag $TARGET_MAG \
 $( [ "$STAIN_NORM" = true ] && echo "--stain_norm" )
