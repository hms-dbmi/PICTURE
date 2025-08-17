## Model inference from a ensemble of N models (defined in mdl_ckpt_files)
##


from Mayo_SSL_model import load_SSL_model
import wandb
from datasets import TileDataset
from models import EnsembleWrapper,  PLWrapper
import numpy as np
from PIL import Image
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import scipy
from pytorch_lightning.callbacks import TQDMProgressBar
from torchvision import transforms
import torch.nn as nn
import torch
import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser
from os.path import basename, dirname
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_auc_score,
)



warnings.filterwarnings('ignore')


# torch and lightning imports

USE_PROG_BAR = True
NUM_WORKERS = 4
PIN_MEMORY = True


parser = ArgumentParser()
# Required arguments
parser.add_argument(
    "--num_classes", help="""Number of classes to be learned.""", type=int, default=2)
parser.add_argument(
    "--num_epochs", help="""Number of Epochs to Run.""", type=int, default=30)
parser.add_argument(
    "--num_folds", help="""Number of Folds to Run.""", type=int, default=10)
parser.add_argument(
    "--k_start", help="""Number of Folds to Run.""", type=int, default=0)
parser.add_argument(
    "--k_end", help="""Number of Folds to Run.""", type=int, default=1)
parser.add_argument("--csv_file", help="""Path to data csv file.""",
                    type=Path, default='GBM_PCNSL_UPenn_FS_repatched_norm_labels.csv')
parser.add_argument("--test_list", help="""Path to test data list (subset of csv_file). set to the None if using the whole csv file. """,
                    type=Path, default=None)
parser.add_argument("-s", "--save_path",
                    help="""Path to save model summary""", default='results_example/')

# Ensemble arguments
parser.add_argument("--ens_dropout", help="""Use dropout for ensembling """,
                    action="store_true", default=False)
parser.add_argument("--ens_aug", help="""Use augmentation for ensembling """,
                    action="store_true", default=False)

# Optional arguments

parser.add_argument("-b", "--batch_size", help="""Manually determine batch size. Defaults to 16.""",
                    type=int, default=16)
parser.add_argument("-p", "--postfix",
                    help="""Postfix for run name""", type=str, default='example')

parser.add_argument("-dg", "--dataset_use_gpus",
                    help="""Enables GPU acceleration.""", action="store_true", default=False)

parser.add_argument(
    "--mdl_ckpt_files", help="""Checkpoint file for trainer""", type=str, nargs='+', \
        default= [
            'trained_models/trained_model_fold0.ckpt',
            'trained_models/trained_model_fold1.ckpt',
            'trained_models/trained_model_fold2.ckpt',
            'trained_models/trained_model_fold3.ckpt',
            'trained_models/trained_model_fold4.ckpt',
        ]
        )


parser.add_argument("--topK", help="""Only Use Top K most confident tiles.""", type=int, default=25)

parser.add_argument(
    "--ens_method",
    help="""method for aggregating across model ensembles""",
    type=str,
    default="median",
    choices=["mean", "median"],
)
parser.add_argument(
    "--agg_method",
    help="""method for aggregating across tiles""",
    type=str,
    default="median",
    choices=["mean", "median", "quantile", "max", "min"],
)
args = parser.parse_args()

LABEL_DICT = {
    'GBM': 0,
    'PCNSL': 1,
}

LABEL_INV_DICT = {0: "GBM", 1: "PCNSL"}



def get_pos_weight(df):
    counts = df['label'].value_counts()
    pos_weight = float(counts[0] / counts[1])
    return pos_weight


def get_transform(ens_aug, ds_size=(224, 224)):
    if ens_aug:
        val_transform = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize(ds_size)
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize(ds_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return val_transform



def filter_topK(df, K=args.topK, col="tile_rank", ascending=True):
    files = list(df["file"])
    folder = np.array([basename(dirname(x)) for x in files])
    basenames = [os.path.basename(x) for x in files]


    uids = np.unique(folder)
    subj_dfs = []
    for id in uids:
        subj_df = df.loc[folder == id]
        subj_df = subj_df.sort_values(by=col, ascending=ascending).reset_index(
            drop=True
        )

        stop_idx = min(K - 1, subj_df.shape[0])
        subj_df = subj_df.loc[:stop_idx]
        subj_dfs.append(subj_df)
        # subj_rows = [df.loc[folder == id].reset_index() for id in uids]
        # subj_rows = [df.sort_values(by='tile_rank').reset_index() for df in subj_rows ]
        # subj_rows =
    df = pd.concat(subj_dfs, ignore_index=True)
    # df = df.loc[df['tile_rank']<=K]
    return df


def get_subj_avg(df, score_col="prob", agg_method="mean", agg_Q=0.5):
    files = list(df["file"])
    folder = np.array([basename(dirname(x)) for x in files])
    uids = np.unique(folder)
    subj_rows = [df.loc[folder == id].reset_index() for id in uids]
    if agg_method == "mean":
        scores = np.array([subj_row[score_col].mean() for subj_row in subj_rows])
    elif agg_method == "max":
        scores = np.array([subj_row[score_col].max() for subj_row in subj_rows])
    elif agg_method == "min":
        scores = np.array([subj_row[score_col].min() for subj_row in subj_rows])
    elif agg_method == "median":
        scores = np.array([subj_row[score_col].median() for subj_row in subj_rows])
    elif agg_method == "quantile":
        scores = np.array(
            [subj_row[score_col].quantile(agg_Q) for subj_row in subj_rows]
        )
    else:
        raise NotImplementedError()
    if "logits" in score_col:
        # scores = 1/(1 + np.exp(-scores))
        scores = scipy.special.expit(scores)

    label = np.array([subj_row["label"].mode()[0] for subj_row in subj_rows])
    label = np.array([subj_row["label"].mode()[0] for subj_row in subj_rows])
    df_out = pd.DataFrame(
        {
            "slide": uids,
            "score": scores,
            "label": label,
        }
    )
    return df_out




def export_performance(y_score, y_label, outpath, prefix="patch_", save_results=True):
    os.makedirs(outpath, exist_ok=True)
    target_names = [LABEL_INV_DICT[i] for i in [0, 1]]
    y_pred = np.where(y_score > 0.5, 1, 0)
    auc = roc_auc_score(y_label, y_score)

    print(classification_report(y_label, y_pred))
    print(f"{prefix}AUC: {auc}")

    CR = classification_report(
        y_label, y_pred, output_dict=True, target_names=target_names
    )
    df = pd.DataFrame.from_dict(CR)
    df.to_csv(os.path.join(outpath, f"{prefix}classification_report.csv"))
    cm = confusion_matrix(y_label, y_pred)
    CR["AUROC"] = auc
    if save_results:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot()
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, f"{prefix}histology_confusion_matrix.jpg"))
        plt.close()

        RocCurveDisplay.from_predictions(y_label, y_score)
        fig = plt.gcf()
        plt.minorticks_on()
        plt.grid(b=True, which="major")
        plt.grid(b=True, which="minor", color="#999999", linestyle=":")
        fig.set_size_inches(10, 10)
        fig.set_dpi(600)

        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f"{prefix}histology ROC curve(AUC=%.3f)" % auc)

        plt.savefig(
            os.path.join(outpath, f"{prefix}histology ROC curve(AUC=%.4f).jpg" % auc),
            dpi=600,
        )
    print(f"saving plots to folder {outpath}")
    return CR


def output_summary(trainer, model, dataloader, csv_filename, dropout=False):
    if dropout:
        model.train()
    else:
        model.eval()
    ## Get prediction file
    results_dict = trainer.predict(model, dataloader)
    # get and save predictions from validation data
    results_all = {}
    for key in results_dict[0].keys():
        val = np.concatenate([x[key] for x in results_dict])
        if val.ndim == 1:
            results_all[key] = val
        else: 
            for c in range(val.shape[1]):
                results_all[f'{key}_{c}'] = val[:, c]
    df = pd.DataFrame(results_all)
    # df['file'] =  data_val['file']
    # save csv
    df.to_csv(csv_filename)
    ## Get subject level summary
    df = filter_topK(df, K=args.topK, col='prob_std_1')
    df_slide = get_subj_avg(
        df, score_col=f"prob_{args.ens_method}_1", agg_method=args.agg_method, agg_Q=0.5)
    df_slide.to_csv(csv_filename.replace('.csv','_slideLevel.csv'))
    
    y_score = df_slide["score"].to_numpy()
    y_label = df_slide["label"].to_numpy()
    Q_str = f"_Q{args.agg_Q:0.2f}" if args.agg_method == "quantile" else ""

    CR_slide = export_performance(
        y_score,
        y_label,
        dirname(csv_filename),
        prefix=f"slide_top{args.topK}_{args.agg_method}Agg{Q_str}_",
    )



def get_model_ensemble(mdl_ckpt_files):
    # takes a list of checkpoints and return a list of models in nn.ModuleList
    model_list = []
    for mdl_ckpt_file in args.mdl_ckpt_files:
        base_model = load_SSL_model()
        checkpoint = torch.load(mdl_ckpt_file, map_location=device)
        base_model.load_state_dict(checkpoint)
        model_list.append(base_model)
    model_list = nn.ModuleList(model_list)
    return model_list
if __name__ == "__main__":

    args_dict = vars(args)
    print('============   parameters   ============')
    for key in args_dict.keys():
        print(f'{key}:\t{args_dict[key]}')
    print('========================================')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    data = pd.read_csv(args.csv_file)
    data = data.reset_index(drop=True)

    for key in LABEL_DICT.keys():
        data['label'].loc[data['label'] == key] = LABEL_DICT[key]
    if args.csv_file != args.test_list and args.test_list is not None :
        test_list = pd.read_csv(args.test_list, header=None)[0]

        data_test = data.loc[data['slide'].isin(test_list)].reset_index()
        # data_train_val = data.loc[~data['slide'].isin(test_list)].reset_index()
    else:
        data_test = data

    val_transform = get_transform(args.ens_aug)
    img_test = TileDataset(
        data_test, transform=val_transform, use_gpu=args.dataset_use_gpus)

    test_dataloader = DataLoader(img_test, batch_size=args.batch_size,
                                 shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    # load model ensembles
    model_list = get_model_ensemble(args.mdl_ckpt_files)
    # model = PLWrapper.load_from_checkpoint(args.mdl_ckpt_file,num_classes = args.num_classes)
    model = EnsembleWrapper(model=model_list, num_classes=args.num_classes,
                      dataset_use_gpus=args.dataset_use_gpus)
    # model.model = base_model

    # Instantiate lightning trainer and train model

    accelerator = "gpu" if device.type == 'cuda' else "cpu"

    trainer_args = {
        'max_epochs': args.num_epochs,
        'enable_progress_bar': USE_PROG_BAR, "accelerator": accelerator, 'auto_scale_batch_size': True,
        'gpus': torch.cuda.device_count(),
        # 'resume_from_checkpoint': args.ckpt_file,
        'callbacks': [TQDMProgressBar(refresh_rate=60)]
    }

    trainer = pl.Trainer(**trainer_args)

    csv_filename = os.path.join(
        args.save_path, f'summary_{args.postfix}.csv'.replace('.ckpt', ''))
    output_summary(trainer, model, test_dataloader,
                   csv_filename, dropout=args.ens_dropout)

