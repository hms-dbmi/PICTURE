import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import GroupShuffleSplit
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import BasePredictionWriter

import torchmetrics as tm
import os
from datetime import datetime
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from PIL import Image
import numpy as np
import h5py
import robusta
from EMA import EMA

USE_PROG_BAR=True   

import torch
from torch.nn.modules.loss import _WeightedLoss


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, resnet_version,
                optimizer='adam', lr=1e-3, 
                decay=1,
                dataset_use_gpus=False,
                class_weight=1,
                transfer=True, tune_fc_only=True):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        #instantiate loss criterion
        self.dataset_use_gpus = dataset_use_gpus
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weight)) \
            if num_classes == 1 \
            else nn.CrossEntropyLoss(weight=torch.tensor([1,class_weight])/(1+class_weight))
        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)
        metrics = tm.MetricCollection(
            {
            'TPR': tm.Recall(task='binary',average='none',num_classes=num_classes),
            'TNR': tm.Specificity(task='binary',average='none',num_classes=num_classes),
            'precision': tm.Precision(task='binary',average='none',num_classes=num_classes),
            'acc': tm.Accuracy(task='binary',num_classes=num_classes),
            'acc_class': tm.Accuracy(task='binary',num_classes=num_classes,average=None),
            'AUROC': tm.AUROC(task='binary',num_classes=num_classes),
            }
        )
        self.metrics = {
            'Train': metrics.clone(prefix='Train/'),
            'Val': metrics.clone(prefix='Val/'),
            'Test': metrics.clone(prefix='Test/')
        }
        self.metrics = nn.ModuleDict(self.metrics)
    
        if tune_fc_only: # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        optimizer =  self.optimizer(self.parameters(), lr=self.lr)
        lr_scheduler = ExponentialLR(optimizer,gamma=self.decay)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
 

    def log_torchmetrics(self,mode):
        metrics = self.metrics[mode].compute()
        for key, val in metrics.items():
            if val.ndim==0:
                self.log(key,val, on_epoch=True)
            else:
                for i in range(val.shape[0]):
                    self.log(f"{key}_{i}",val[i], on_epoch=True)
        return metrics
    def reset_torchmetrics(self,mode):
        self.metrics[mode].reset()
        return
    def training_step(self, batch, batch_idx):
        x, y,_ = batch
        # y = y.float()
        logits = self(x).squeeze()
        if y.shape[0]==1:
            logits = logits.unsqueeze(0)
        if logits.ndim==1:
            y = y.float()

        prob = F.sigmoid(logits)
        loss = self.criterion(logits, y)
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=USE_PROG_BAR, logger=True)

        self.metrics['Train'].update(logits, y.int())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y,_ = batch
        # y = y.float()
        logits = self(x).squeeze()
        if y.shape[0]==1:
            logits = logits.unsqueeze(0)
        if logits.ndim==1:
            y = y.float()

        # prob = F.sigmoid(logits)
        loss = self.criterion(logits, y)
        self.log("valid_loss", loss, on_step=True, prog_bar=USE_PROG_BAR, logger=True)
        self.metrics['Val'].update(logits, y.int())


    def test_step(self, batch, batch_idx):
        x, y,_ = batch
        logits = self(x).squeeze()
        if y.shape[0]==1:
            logits = logits.unsqueeze(0)
        if logits.ndim==1:
            y = y.float()
        # prob = F.sigmoid(logits)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss, on_step=True, prog_bar=USE_PROG_BAR, logger=True)
        self.metrics['Test'].update(logits, y.int())

    def predict_step(self, batch, batch_idx):
        x, y,file = batch
        # y = y.float()
        logits = self(x).squeeze()
        if y.shape[0]==1:
            logits = logits.unsqueeze(0)
        if logits.ndim==1:
            y = y.float()
            prob = F.sigmoid(logits)
        else: 
            prob = F.softmax(logits,dim=1)

        pred_dict = {
            'file': np.array(file),
            'prob':prob.cpu().numpy(),
            'logits':logits.cpu().numpy(),
            'label':y.cpu().numpy()}
        return pred_dict 

    def on_train_epoch_end(self):
        # output = self.metrics['Train'].compute()
        # self.log_dict(output)
        output = self.log_torchmetrics('Train')
        self.reset_torchmetrics('Train')
    def on_validation_epoch_end(self):
        # output = self.metrics['Val'].compute()
        output = self.log_torchmetrics('Val')
        self.reset_torchmetrics('Val')
    def test_epoch_end(self,patient_result):
        # output = self.metrics['Test'].compute()
        output = self.log_torchmetrics('Test')
        self.reset_torchmetrics('Test')
        return (patient_result, output)

class PLWrapper(pl.LightningModule):
    def __init__(self, num_classes,
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
                class_weight=1):
        super().__init__()
        self.__dict__.update(locals())
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        #instantiate loss criterion
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
                updates_every= self.EMA_update_every,
                ens_mode=self.EMA_mode)
    
    def initialize_loss(self):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight)) \
            if self.num_classes == 1 \
            else nn.CrossEntropyLoss(weight=torch.tensor([1,self.class_weight])/(1+self.class_weight))
    def initialize_metrics(self):
        metrics = tm.MetricCollection(
            {
            'TPR': tm.Recall(task='multiclass',average='none',num_classes=self.num_classes),
            'TNR': tm.Specificity(task='multiclass',average='none',num_classes=self.num_classes),
            'precision': tm.Precision(task='multiclass',average='none',num_classes=self.num_classes),
            'acc': tm.Accuracy(task='multiclass',num_classes=self.num_classes),
            'acc_class': tm.Accuracy(task='multiclass',num_classes=self.num_classes,average=None),
            # 'AUROC': tm.AUROC(task='binary',num_classes=self.num_classes),
            'AUROC': tm.AUROC(task='multiclass',num_classes=self.num_classes),
            }
        )
        self.metrics = {
            'Train': metrics.clone(prefix='Train/'),
            'Val': metrics.clone(prefix='Val/'),
            'Test': metrics.clone(prefix='Test/')
        }
        self.metrics = nn.ModuleDict(self.metrics)
    def forward(self, X):
        return self.model(X)
        # if self.training or not self.use_EMA:
        #     return self.model(X)
        # else:
        #     return self.EMA_model(X)
    def configure_optimizers(self):
        optimizer =  self.optimizer(self.model.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        lr_scheduler = ExponentialLR(optimizer,gamma=self.decay)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
 

    def log_torchmetrics(self,mode):
        metrics = self.metrics[mode].compute()
        for key, val in metrics.items():
            if val.ndim==0:
                self.log(key,val, on_epoch=True)
            else:
                for i in range(val.shape[0]):
                    self.log(f"{key}_{i}",val[i], on_epoch=True)
        return metrics
    def reset_torchmetrics(self,mode):
        self.metrics[mode].reset()
        return
    def training_step(self, batch, batch_idx):
        x, y,_ = batch
        # y = y.float()
        logits = self(x).squeeze()
        if y.shape[0]==1:
            logits = logits.unsqueeze(0)
        if logits.ndim==1:
            y = y.float()
            prob = F.sigmoid(logits)
        else: 
            prob = F.softmax(logits,dim=1)
        # prob = F.sigmoid(logits)
        loss = self.criterion(logits, y)
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=USE_PROG_BAR, logger=True)
        self.metrics['Train'].update(logits, y.int())
        if self.use_EMA:
            self.model.update()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y,_ = batch
        # y = y.float()
        logits = self(x).squeeze()
        if y.shape[0]==1:
            logits = logits.unsqueeze(0)
        if logits.ndim==1:
            y = y.float()
            prob = F.sigmoid(logits)
        else: 
            prob = F.softmax(logits,dim=1)
        # prob = F.sigmoid(logits)
        loss = self.criterion(logits, y)
        self.log("valid_loss", loss, on_step=True, prog_bar=USE_PROG_BAR, logger=True)
        self.metrics['Val'].update(logits, y.int())


    def test_step(self, batch, batch_idx):
        x, y,_ = batch
        # y = y.float()
        logits = self(x).squeeze()
        if y.shape[0]==1:
            logits = logits.unsqueeze(0)
        if logits.ndim==1:
            y = y.float()
        #prob = F.sigmoid(logits)
        if logits.ndim==1:
            y = y.float()
            prob = F.sigmoid(logits)
        else: 
            prob = F.softmax(logits,dim=1)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss, on_step=True, prog_bar=USE_PROG_BAR, logger=True)
        self.metrics['Test'].update(logits, y.int())

    def predict_step(self, batch, batch_idx):
        self.model.eval()
        x, y,file = batch
        # y = y.float()
        logits = self(x).squeeze()
        if y.shape[0]==1:
            logits = logits.unsqueeze(0)
        # if logits.ndim==1:
        #     y = y.float()
        # prob = F.sigmoid(logits)

        if logits.ndim==1:
            y = y.float()
            prob = F.sigmoid(logits)
        else:
            prob = F.softmax(logits,dim=-1)

        pred_dict = {
            'file': np.array(file),
            'prob':prob.detach().cpu().numpy(),
            'logits':logits.detach().cpu().numpy(),
            'label':y.detach().cpu().numpy()}
        return pred_dict 

    def on_train_epoch_end(self):
        # output = self.metrics['Train'].compute()
        # self.log_dict(output)
        output = self.log_torchmetrics('Train')
        self.reset_torchmetrics('Train')
    def on_validation_epoch_end(self):
        # output = self.metrics['Val'].compute()
        output = self.log_torchmetrics('Val')
        self.reset_torchmetrics('Val')
    def test_epoch_end(self,patient_result):
        # output = self.metrics['Test'].compute()
        output = self.log_torchmetrics('Test')
        self.reset_torchmetrics('Test')
        return (patient_result, output)


class CustomWriter(BasePredictionWriter):

    def __init__(self, hd5_file, write_interval):
        super().__init__(write_interval)
        self.hd5_file = hd5_file
        self.hf = h5py.File(hd5_file, 'w')

    def write_on_batch_end(self, trainer, pl_module, predictions, batch_indices, batch, batch_idx, dataloader_idx):
        keys = list(predictions.keys())
        for key in keys:
            data =  predictions[key]
            # if data.dtype == '<U164': # string  arrays need encoding
            if  '<U' in str(data.dtype): # string  arrays need encoding
                utf8_type = h5py.string_dtype('utf-8', 512)
                data = list(data)
                data = [np.array(x.encode("utf-8"), dtype=utf8_type) for x in data]
            if key not in self.hf:
                maxshape = list(predictions[key].shape)
                maxshape[0] = None
                self.hf.create_dataset(key, data=data, compression="gzip", chunks=True, maxshape=maxshape)
            else:
                self.hf[key].resize((self.hf[key].shape[0] + predictions[key].shape[0]), axis=0)
                self.hf[key][-predictions[key].shape[0]:] = data

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        for key in keys:
            self.hf.create_dataset(key, data=predictions[key], compression="gzip")
    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        self.hf.close()

class EncoderWrapper(PLWrapper):
    def forward(self, X):
        features = self.model[0](X)
        output = self.model[1](features)
        return features, output

    def predict_step(self, batch, batch_idx):
        self.model.eval()
        x, y,file = batch
        # y = y.float()
        features, logits = self(x)
        logits = logits.squeeze()
        if y.shape[0]==1:
            logits = logits.unsqueeze(0)
        # if logits.ndim==1:
        #     y = y.float()
        # prob = F.sigmoid(logits)

        if logits.ndim==1:
            y = y.float()
            prob = F.sigmoid(logits)
        else:
            prob = F.softmax(logits,dim=-1)

        pred_dict = {
            'file': np.array(file),
            'prob':prob.cpu().numpy(),
            'features': features.cpu().numpy(),
            'logits':logits.cpu().numpy(),
            'label':y.cpu().numpy()}
        return pred_dict 

             


class EnsembleWrapper(PLWrapper):

    def forward(self, X, y):
        logits_ens = [model(X) for model in self.model]
        if y.shape[0]==1:
            logits_ens = [logits.unsqueeze(0) for logits in logits_ens]
        
        logits_ens = torch.stack(logits_ens,dim=0)
        return logits_ens

    def predict_step(self, batch, batch_idx):
        x, y,file = batch
        # logits_ens = [self(x).squeeze() for i in range(self.n_ens)]
        # if y.shape[0]==1:
        #     logits_ens = [logits.unsqueeze(0) for logits in logits_ens]
        
        # logits_ens = torch.stack(logits_ens,dim=0)
        logits_ens = self(x,y)
        if logits_ens.ndim==2:
            y = y.float()
            prob_ens = F.sigmoid(logits_ens)
        else:
            prob_ens = F.softmax(logits_ens,dim=-1)

        # if logits_ens.ndim==2:
        #     y = y.float()
        # prob_ens = F.sigmoid(logits_ens)
        
        logits_mean = torch.mean(logits_ens,dim=0)
        logits_median = torch.median(logits_ens,dim=0)[0]
        logits_std = torch.std(logits_ens,dim=0)

        prob_mean = torch.mean(prob_ens,dim=0)
        prob_median = torch.median(prob_ens,dim=0)[0]
        prob_std = torch.std(prob_ens,dim=0)


        pred_dict = {
            'file': np.array(file),
            'logits_mean':logits_mean.cpu().numpy(),
            'logits_median':logits_median.cpu().numpy(),
            'logits_std':logits_std.cpu().numpy(),
            'prob_mean':prob_mean.cpu().numpy(),
            'prob_median':prob_median.cpu().numpy(),
            'prob_std':prob_std.cpu().numpy(),
            'label':y.cpu().numpy()}
        return pred_dict 


     

class EnsembleWrapperAlt(EnsembleWrapper):
    ##
    def predict_step(self, batch, batch_idx):
        x, y,file = batch
        # y = y.float()
        if self.ens_dropout:
            self.model.train()
        else:
            self.model.eval()
        logits_ens = [self(x).squeeze() for i in range(self.n_ens)]
        if y.shape[0]==1:
            logits_ens = [logits.unsqueeze(0) for logits in logits_ens]
        
        logits_ens = torch.stack(logits_ens,dim=0)
        if logits_ens.ndim==2:
            y = y.float()
            prob_ens = F.sigmoid(logits_ens)
        else:
            prob_ens = F.softmax(logits_ens,dim=-1)

        # if logits_ens.ndim==2:
        #     y = y.float()
        # prob_ens = F.sigmoid(logits_ens)

        logits_mean = torch.mean(logits_ens,dim=0)
        logits_median = torch.median(logits_ens,dim=0)[0]
        logits_std = torch.std(logits_ens,dim=0)

        prob_mean = torch.mean(prob_ens,dim=0)
        prob_median = torch.median(prob_ens,dim=0)[0]
        prob_std = torch.std(prob_ens,dim=0)


        pred_dict = {
            'file': np.array(file),
            'logits_mean':logits_mean.cpu().numpy(),
            'logits_median':logits_median.cpu().numpy(),
            'logits_std':logits_std.cpu().numpy(),
            'prob_mean':prob_mean.cpu().numpy(),
            'prob_median':prob_median.cpu().numpy(),
            'prob_std':prob_std.cpu().numpy(),
            'label':y.cpu().numpy()}
        return pred_dict 


