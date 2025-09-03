import pandas as pd
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor

def select_random_n_tiles(df, n=3):
    """
    Randomly select N tiles for each slide.

    Parameters:
    - df: DataFrame containing the tiles information.
    - n: Number of tiles to randomly select.

    Returns:
    - DataFrame with N randomly selected tiles for each slide.
    """
    return df.groupby('slide').apply(lambda x: x.sample(n=min(len(x), n))).reset_index(drop=True)

def main(anno_path):
    # Load model output from a CSV file
    GvL_model_output = pd.read_csv(anno_path, index_col=0)

    # Extract the filename without the extension to use as a save key
    save_key = anno_path.split('/')[-1].replace('.csv','')

    # Select high confidence predictions for L and G
    High_L_TP = GvL_model_output[GvL_model_output['label'] == 1]
    High_G_TP = GvL_model_output[GvL_model_output['label'] == 0]

    # Randomly select N tiles for both High_L_TP and High_G_TP
    High_L_TP_selected = select_random_n_tiles(High_L_TP, n=300)
    High_G_TP_selected = select_random_n_tiles(High_G_TP, n=300)

    # Initialize the NucleusInstanceSegmentor
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        num_loader_workers=8,
        num_postproc_workers=8,
        batch_size=1000,
    )

    # Predict on selected tiles for G
    inst_segmentor.predict(
        High_G_TP_selected['file'].to_list(),
        save_dir=f"placeholder_directory/{save_key}_GBM_control_more+/",
        mode="tile",
        on_gpu=True,
        crash_on_exception=True,
    )



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Script for nuclei segmentation and feature extraction.')
    parser.add_argument('--anno_path', required=True, help='Path to the annotation CSV file.')

    args = parser.parse_args()

    main(args.anno_path)
    print('Feature extraction completed.')
