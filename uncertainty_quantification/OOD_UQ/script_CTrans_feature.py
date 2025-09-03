import argparse
import torch
from torch.optim import AdamW
from pytorch_lightning import Trainer
from src.datasets.ood_dataset import OodDataset
# from src.datamodules.vienna_datamodule import ViennaDataModule
# from src.datamodules.ebrain_datamodule import EbrainDataModule
from src.datamodules.ebrain_feature_datamodule import EbrainFeatureDataModule
from src.datamodules.vienna_feature_datamodule import ViennaFeatureDataModule
from src.models.uncertainty_module import UncertaintyModuleCTransFeatures
from src.models.slides_module import ViT8
import torch.multiprocessing
from omegaconf import DictConfig
import json
from datetime import datetime

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
import os
import pandas as pd

CKPT_PATH='ckpt_debug'
USE_PROG_BAR=True
NUM_WORKERS=0
PIN_MEMORY=True

torch.multiprocessing.set_sharing_strategy("file_system")

# Define argparse and argument for checkpoint path
parser = argparse.ArgumentParser()
parser.add_argument(
    "--latent_dim",
    default=None,
    type=int,
    help="latent_dim",
)
parser.add_argument(
    "--max_epochs",
    default=50,
    type=int,
    help="max_epochs",
)

parser.add_argument(
    "--check_val_every_n_epoch",
    default=1,
    type=int,
    help="check_val_every_n_epoch",
)
parser.add_argument(
    "--lr",
    default=0.001,
    type=float,
    help="learning rate",
)
parser.add_argument(
    "--batch_size",
    default=500,
    type=int,
    help="batch_size",
)

parser.add_argument(
    "--radial_flow_layers",
    default=2,
    type=int,
    help="radial_flow_layers",
)
parser.add_argument(
    "--resume_checkpoint",
    help="Resume from checkpoint",
    # default="ckpt_debug/debug_20231228_202844/GBM_PCNSL_OOD/20p6ey03/checkpoints/epoch=4-step=105.ckpt",
    type=str,
    default=None,
) 


parser.add_argument(
    "--extra_OOD",
    help="Use extra OOD",
    action='store_true',
    default=True,
) 
parser.add_argument(
    "--no_extra_OOD",
    help="Don't ese extra OOD",
    action='store_false',
    dest='extra_OOD'
) 


args = parser.parse_args()

args_dict = vars(args)
print('============   parameters   ============')
for key in args_dict.keys():
    print(f'{key}:\t{args_dict[key]}')
print('========================================')


device=  torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:\t{device}")

# Load model checkpoint
optim_config=DictConfig

# Initialize Vienna data module
vienna_datamodule = ViennaFeatureDataModule(
    batch_size=args.batch_size, num_workers=NUM_WORKERS,    class_balancing=True)
vienna_datamodule.setup()
train_loader = vienna_datamodule.train_dataloader()


model = UncertaintyModuleCTransFeatures(
    latent_dim=args.latent_dim,
    number_of_classes=2,
    optimizer=torch.optim.Adam,
    lr=args.lr,
    device=device,
    locked_encoder=True,
    locked_classifier=False,
    radial_flow_layers=args.radial_flow_layers,
    extra_ood=args.extra_OOD,
    batch_size=args.batch_size)


# Initialize Ebrain data module with extra OOD data
# # ood_datamodule = EbrainDataModule(batch_size=args.batch_size, num_workers=4, extra_ood=True)
# ood_datamodule = EbrainDataModule(batch_size=args.batch_size, num_workers=4, extra_ood=args.extra_OOD)
# ood_datamodule.setup()
# model.ood_datamodule = ood_datamodule


os.makedirs(CKPT_PATH,exist_ok=True)


now = datetime.now() # current date and time
date_time = now.strftime("%Y%m%d_%H%M%S")
run_name = f'debug_{date_time}'
save_path =  os.path.join(CKPT_PATH,run_name)
if not os.path.isdir(save_path):
    os.makedirs(save_path)

with open(os.path.join(save_path,"args.json"), "w") as outfile:
    json.dump(vars(args),outfile)

# wandb_logger = WandbLogger(name=run_name,project='GBM_PCNSL_OOD',save_dir=save_path)
accelerator = "gpu" if device.type == 'cuda' else "cpu"
trainer_args = {
    'max_epochs': args.max_epochs,
    # 'logger': wandb_logger,
    'enable_progress_bar': USE_PROG_BAR,"accelerator":accelerator,'auto_scale_batch_size':True,
    'check_val_every_n_epoch': args.check_val_every_n_epoch,
    'gpus': torch.cuda.device_count(),
    'callbacks':[TQDMProgressBar(refresh_rate=10)],
}

trainer = Trainer(**trainer_args)

# Evaluate model performance for different seeds
# scaler = torch.cuda.amp.GradScaler()
# model.to("cuda:0")

for seed in [0]:
    # Load in-domain (ID) dataset from Vienna data module
    # id_dataset = vienna_datamodule.val_dataloader().dataset
    model.seed = seed
    model.hparams.seed = seed
    # trainer.validate(model, vienna_datamodule)

    trainer.fit(model, vienna_datamodule,ckpt_path=args.resume_checkpoint)
    train_metrics = trainer.callback_metrics

    # trainer.validate(model, vienna_datamodule)
    trainer.test(model, vienna_datamodule)
    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    # df = pd.DataFrame(metric_dict)
    df = pd.Series(metric_dict)
    df.to_csv(os.path.join(CKPT_PATH,'results.csv'))


    # return metric_dict, object_dict
