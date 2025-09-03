import wandb
from typing import Literal
import copy
import h5py
from EMA import EMA
import numpy as np
from PIL import Image
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datetime import datetime
import os
import torchmetrics as tm
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import GroupShuffleSplit
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.optim import SGD, Adam
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser

from torch import Tensor

warnings.filterwarnings('ignore')

# torch and lightning imports

# from ema_pytorch import EMA


USE_PROG_BAR = True


def MLP(in_features, fc_latent_size, num_classes, dropout=0.5):

    if fc_latent_size is None:
        fc_latent_size = []
    # MLP
    fc_ins = [in_features] + fc_latent_size
    fc_outs = fc_latent_size + [num_classes]
    # replace final layer for fine tuning
    fc_list = []
    for i in range(len(fc_ins)-1):
        fc_in = fc_ins[i]
        fc_out = fc_outs[i]
        if dropout is None:
            fc = nn.Sequential(
                nn.Linear(fc_in, fc_out),
                nn.ReLU()
            )
        else:
            fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(fc_in, fc_out),
                nn.ReLU()
            )
        fc_list.append(fc)
    fc_in = fc_ins[-1]
    fc_out = fc_outs[-1]
    if dropout is None:
        fc = nn.Linear(fc_in, fc_out)
    else:
        fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fc_in, fc_out),
        )
    fc_list.append(fc)
    # if num_classes == 1:
    #     fc_list.append(nn.Sigmoid())
    model = nn.Sequential(*fc_list)
    return model



def CNN(
        kernel_size,
        in_features, 
        fc_latent_size,
        num_classes,
        dropout=0.5,
        conv_fn = nn.Conv1d,
        batch_norm_fn=None,
        **kwargs):
    if fc_latent_size is None:
        fc_latent_size = []
    # MLP
    fc_ins = [in_features] + fc_latent_size
    fc_outs = fc_latent_size + [num_classes]
    # replace final layer for fine tuning
    fc_list = []
    for i in range(len(fc_ins)-1):
        fc_in = fc_ins[i]
        fc_out = fc_outs[i]
        fc = nn.Sequential(
            nn.Dropout(p=dropout) if dropout is not None else nn.Identity(), 
            conv_fn(fc_in, fc_out,kernel_size=kernel_size,**kwargs),
            nn.ReLU(),
            batch_norm_fn(fc_out) if batch_norm_fn is not None else nn.Identity()
        )
        fc_list.append(fc)
    fc = nn.Sequential(
        nn.Dropout(p=dropout) if dropout is not None else nn.Identity(), 
        conv_fn(fc_ins[-1], fc_outs[-1],kernel_size=kernel_size,**kwargs)
    )
    fc_list.append(fc)
    # if num_classes == 1:
    #     fc_list.append(nn.Sigmoid())
    model = nn.Sequential(*fc_list)
    return model


class MILAttention(nn.Module):
    def __init__(self, featureLength = 512, featureInside = 256):
        '''
        Parameters:
            featureLength: Length of feature passed in from feature extractor(encoder)
            featureInside: Length of feature in MIL linear
        Output: tensor
            weight of the features
        '''
        super(MILAttention, self).__init__()
        self.featureLength = featureLength
        self.featureInside = featureInside

        self.attetion_V = nn.Sequential(
            nn.Linear(self.featureLength, self.featureInside, bias = True),
            nn.Tanh()
        )
        self.attetion_U = nn.Sequential(
            nn.Linear(self.featureLength, self.featureInside, bias = True),
            nn.Sigmoid()
        )
        self.attetion_weights = nn.Linear(self.featureInside, 1, bias = True)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x: Tensor) -> Tensor:
        bz, pz, fz = x.shape
        x = x.view(bz*pz, fz)
        att_v = self.attetion_V(x)
        att_u = self.attetion_U(x)
        att = self.attetion_weights(att_u*att_v)
        weight = att.view(bz, pz, 1)
        
        weight = self.softmax(weight)
        weight = weight.view(bz, 1, pz)

        return weight

class MILNet(nn.Module):
    def __init__(self,in_features,fc_latent_size,MIL_latent_size,num_classes=1,dropout=0.5):
        super().__init__()

        self.classifier = MLP(
                in_features=in_features,
                fc_latent_size=fc_latent_size,
                num_classes=1,
                dropout= dropout)
        self.attention = MILAttention(
                featureLength = in_features, featureInside = MIL_latent_size)
    def forward(self,x):
        weight = self.attention(x)
        features_MIL = torch.bmm(weight, x).squeeze(1)
        out = self.classifier(features_MIL)
        return out

class CustomWriter(BasePredictionWriter):

    def __init__(self, hd5_file, write_interval):
        super().__init__(write_interval)
        self.hd5_file = hd5_file
        self.hf = h5py.File(hd5_file, 'w')

    def write_on_batch_end(self, trainer, pl_module, predictions, batch_indices, batch, batch_idx, dataloader_idx):
        keys = list(predictions.keys())
        for key in keys:
            data = predictions[key]
            # if data.dtype == '<U164': # string  arrays need encoding
            if '<U' in str(data.dtype):  # string  arrays need encoding
                utf8_type = h5py.string_dtype('utf-8', 512)
                data = list(data)
                data = [np.array(x.encode("utf-8"), dtype=utf8_type)
                        for x in data]
            if key not in self.hf:
                maxshape = list(predictions[key].shape)
                maxshape[0] = None
                self.hf.create_dataset(
                    key, data=data, compression="gzip", chunks=True, maxshape=maxshape)
            else:
                self.hf[key].resize(
                    (self.hf[key].shape[0] + predictions[key].shape[0]), axis=0)
                self.hf[key][-predictions[key].shape[0]:] = data

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        for key in keys:
            self.hf.create_dataset(
                key, data=predictions[key], compression="gzip")

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        self.hf.close()


class PLWrapper(pl.LightningModule):
    def __init__(self, num_classes=1,
                 model=None,
                 use_EMA=False,
                 EMA_beta=0.999,
                 EMA_update_after_step=1000,
                 EMA_update_every=100,
                 EMA_mode='exp',
                 optimizer='adam', lr=1e-3,
                 weight_decay=0,
                 decay=1,
                 dataset_use_gpus=False,
                 class_weight=1, **kwargs):
        super().__init__()
        self.__dict__.update(**kwargs)
        self.__dict__.update(locals())
        self.__dict__.pop('kwargs')
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # instantiate loss criterion
        self.initialize_model()
        self.initialize_EMA()
        self.initialize_loss()
        self.initialize_metrics()

    def initialize_model(self):
        if self.model is None:
            self.model = nn.Sequential()

    def initialize_EMA(self):
        if self.use_EMA:
            self.model = EMA(
                self.model,
                decay=self.EMA_beta,
                starts_at=self.EMA_update_after_step,
                updates_every=self.EMA_update_every,
                ens_mode=self.EMA_mode)

    def initialize_loss(self):
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight)) \
        #     if self.num_classes == 1 \
        #     else nn.CrossEntropyLoss(weight=torch.tensor([1, self.class_weight])/(1+self.class_weight))
        
        if self.class_weight is None:
            class_weight=None
        else:
            class_weight = torch.tensor(self.class_weight) \
            if self.num_classes == 1 \
            else torch.tensor([1, self.class_weight])/(1+self.class_weight)
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight,reduction='none') \
            if self.num_classes == 1 \
            else nn.CrossEntropyLoss(weight=class_weight,reduction='none')

    def initialize_metrics(self):
        metrics = tm.MetricCollection(
            {
                'TPR': tm.Recall(task='binary', average='none', num_classes=self.num_classes),
                'TNR': tm.Specificity(task='binary', average='none', num_classes=self.num_classes),
                'precision': tm.Precision(task='binary', average='none', num_classes=self.num_classes),
                'acc': tm.Accuracy(task='binary', num_classes=self.num_classes),
                'acc_class': tm.Accuracy(task='binary', num_classes=self.num_classes, average=None),
                'AUROC': tm.AUROC(task='binary', num_classes=self.num_classes)
            }
        )
        self.metrics = {
            'Train': metrics.clone(prefix='Train/'),
            'Val': metrics.clone(prefix='Val/'),
            'Test': metrics.clone(prefix='Test/')
        }
        self.metrics = nn.ModuleDict(self.metrics)

    def forward(self, X, *args,**kwargs):
        return self.model(X, *args,**kwargs)
        # if self.training or not self.use_EMA:
        #     return self.model(X)
        # else:
        #     return self.EMA_model(X)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = ExponentialLR(optimizer, gamma=self.decay)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def log_torchmetrics(self, mode):
        metrics = self.metrics[mode].compute()
        for key, val in metrics.items():
            if isinstance(val, wandb.viz.CustomChart):
                # self.log(key,val, on_epoch=True)
                wandb.log({key: val})
            elif isinstance(val, pd.DataFrame):
                # self.log(key,val, on_epoch=True)
                # self.logger.log_table(f'{key}/Epoch{self.current_epoch}',dataframe=val)
                self.logger.log_table(key, dataframe=val,
                                      step=self.current_epoch)

            elif val.ndim == 0:
                self.log(key, val, on_epoch=True)
            else:
                for i in range(val.shape[0]):
                    self.log(f"{key}_{i}", val[i], on_epoch=True)
        return metrics

    def reset_torchmetrics(self, mode):
        self.metrics[mode].reset()
        return

    @staticmethod
    def result_dict_to_numpy(results, exclude=[]):  # convert results to numpy array
        for key in exclude:
            if key in results.keys():
                results.pop(key)
        for key in results.keys():
            if isinstance(results[key], torch.Tensor):
                results[key] = results[key].cpu().numpy()
            elif isinstance(results[key], (list, tuple)):
                results[key] = np.array(results[key])
        return results
    # inferences

    def batch_inference(self, batch, batch_idx, **kwargs):
        # x, label, file = batch
        x, label, file = batch["image"], batch["label"], batch["patient_id"]

        # label = label.float()
        logits = self.forward(x, **kwargs).squeeze()
        if label.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if logits.ndim == label.ndim:
            label = label.float()
            prob = F.sigmoid(logits)
        else:
            prob = F.softmax(logits, dim=-1)
        loss = self.criterion(logits, label)
        results = {
            'label': label,
            'logits': logits,
            'prob': prob,
            'loss': torch.mean(loss),
            'file': file,
        }
        return results

    def training_step(self, batch, batch_idx):
        # perform logging
        results = self.batch_inference(batch, batch_idx)

        self.log("train_loss", results['loss'], on_step=True,
                 on_epoch=True, prog_bar=USE_PROG_BAR, logger=True)
        self.metrics['Train'].update(results['logits'], results['label'].int())
        if self.use_EMA:
            self.model.update()
        return results['loss']

    def validation_step(self, batch, batch_idx):
        results = self.batch_inference(batch, batch_idx)

        self.log("valid_loss", results['loss'],
                 on_step=True, prog_bar=USE_PROG_BAR, logger=True)
        self.metrics['Val'].update(results['logits'], results['label'].int())

    def test_step(self, batch, batch_idx):
        results = self.batch_inference(batch, batch_idx)

        self.log("test_loss", results['loss'],
                 on_step=True, prog_bar=USE_PROG_BAR, logger=True)
        self.metrics['Test'].update(results['logits'], results['label'].int())

    def predict_step(self, batch, batch_idx):
        results = self.batch_inference(batch, batch_idx)
        results = self.result_dict_to_numpy(results, exclude=['loss'])
        return results

    def on_train_epoch_end(self):
        # output = self.metrics['Train'].compute()
        # self.log_dict(output)
        output = self.log_torchmetrics('Train')
        self.reset_torchmetrics('Train')

    def on_validation_epoch_end(self):
        # output = self.metrics['Val'].compute()
        output = self.log_torchmetrics('Val')
        self.reset_torchmetrics('Val')

    def test_epoch_end(self, patient_result):
        # output = self.metrics['Test'].compute()
        output = self.log_torchmetrics('Test')
        self.reset_torchmetrics('Test')
        return (patient_result, output)


class MLPClassifier(PLWrapper):
    def __init__(self, *args,
                 dropout=0.5,
                 in_features,
                 fc_latent_size=[],
                 **kwargs):
        self.__dict__.update(locals())
        super().__init__(*args, **kwargs)

    def initialize_model(self):
        self.model = MLP(
            self.in_features, self.fc_latent_size, num_classes=1, dropout=self.dropout)

