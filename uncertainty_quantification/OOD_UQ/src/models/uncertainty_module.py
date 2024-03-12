from typing import Any, List, Literal
from matplotlib import cm, colorbar

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, AUROC
from torchmetrics.classification.accuracy import Accuracy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from src.models.slides_module import ViT8
from src.models.Mayo_SSL_Uncertainty import loadCTransEncoder, loadCTransClassifier
# from Mayo_SSL_Uncertainty import loadCTransEncoder
from torch.utils.data import ConcatDataset

from sklearn.preprocessing import MinMaxScaler

from typing import Any, List
import torch
from src.models.natpn.metrics import BrierScore, AUCPR
from src.models.natpn.nn import BayesianLoss, NaturalPosteriorNetworkModel
from src.models.natpn.nn.output import CategoricalOutput
from src.models.natpn.nn.flow.radial import RadialFlow
from src.models.natpn.nn.flow._base import NormalizingFlow
from src.models.natpn.nn.flow.transforms import RadialTransform
from src.models.natpn.nn.flow.maf import MaskedAutoregressiveFlow
from src.models.natpn.nn.model import NaturalPosteriorNetworkModel
from src.models.natpn.nn.scaler import CertaintyBudget
import wandb

from tqdm import tqdm
from src.datamodules.cns_datamodule import CNSDataModule
from src.datamodules.ebrain_datamodule import EbrainDataModule
from src.datamodules.ebrain_feature_datamodule import EbrainFeatureDataModule
from src.datasets.ood_dataset import OodDataset

import random
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch import nn
import torch.nn.functional as F
import os
import multiprocessing

FLOW_TYPE_DICT = {
    "radial": RadialFlow,
    "MAF": MaskedAutoregressiveFlow,
}

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


class UncertaintyModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer = None,
        net: torch.nn.Module = ViT8,
        exclude_uncertain: bool = False,
        latent_dim: int = 16,
        radial_flow_layers: int = 6,
        flow_type: Literal["radial","MAF"] = "radial",
        certainty_budget: CertaintyBudget = "normal",
        entropy_weight: float = 1e-5,
        number_of_classes: int = 5,
        binary_label: int = None,
        clamp_scaling: bool = True,
        locked_encoder: bool= False,
        cancer_ood: bool=True,
        extra_ood: bool=False,
        seed: int = 0,
        batch_size=1000,
        output_val_tsne=False,
        lr=0.001,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        ood_datamodule = EbrainDataModule(
            batch_size=batch_size, num_workers=4, cancer_ood=cancer_ood, extra_ood=extra_ood, exclude_uncertain=exclude_uncertain)
        ood_datamodule.setup()
        self.ood_datamodule = ood_datamodule
        flow_class = FLOW_TYPE_DICT[self.hparams['flow_type']]
        self.net = NaturalPosteriorNetworkModel(
            latent_dim = self.hparams["latent_dim"],
            clamp_scaling = self.hparams["clamp_scaling"],
            encoder = net(num_class=self.hparams["latent_dim"], locked_encoder=self.hparams["locked_encoder"]),
            flow = flow_class(self.hparams["latent_dim"], num_layers=self.hparams["radial_flow_layers"]),
            certainty_budget=self.hparams["certainty_budget"],
            output = CategoricalOutput(self.hparams["latent_dim"], self.hparams["number_of_classes"]))
        
        # loss function
        self.criterion = BayesianLoss(entropy_weight=self.hparams["entropy_weight"])

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy(mdmc_reduce="global")
        self.val_acc = Accuracy(mdmc_reduce="global")
        self.test_acc = Accuracy(mdmc_reduce="global")
        self.train_auroc = AUROC()
        self.val_auroc = AUROC()
        self.test_auroc = AUROC()




        # We have discrete output
        self.output = "discrete"
        self.brier_score = BrierScore(compute_on_step=False, dist_sync_fn=self.all_gather, full_state_update=False)

        self.alea_conf_pr = AUCPR(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.alea_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

        self.epist_conf_pr = AUCPR(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.epist_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()




    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch["image"], batch["label"]
        preds, log_prob = self.forward(x)
        loss = self.criterion(preds, y)
        y_logits = preds.maximum_a_posteriori().logits
        y_prob = F.softmax(y_logits,dim=-1)
        y_pred = preds.maximum_a_posteriori().mean()
        y_pred_latent = preds.maximum_a_posteriori().expected_sufficient_statistics()

        if self.hparams.binary_label:
            y_pred = (y_pred == self.hparams.binary_label).int()

        return loss, y_prob, y_pred, y_pred_latent, y, log_prob

    def training_step(self, batch: Any, batch_idx: int):
        loss, y_prob, y_pred, y_pred_latent, targets, log_prob = self.step(batch)

        # log train metrics
        acc = self.train_acc(y_pred, targets)
        self.train_auroc.update(y_prob[:,1],targets)

        brier_score = self.brier_score(y_pred_latent, targets)
        aleatoric_confidence = self.net.aleatoric_confidence(batch["image"])
        epistemic_confidence = self.net.epistemic_confidence(batch["image"])

        # log calibration metrics
        # We try to maximize the prediction confidence (epistemic_confidence) of the model while minimizing the calibration error (brier_score).
        # Hence we maximize the calibration metric.
        calibration_metric = 0.33*epistemic_confidence.mean() - 0.33*brier_score + 0.33*aleatoric_confidence.mean()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)        
        self.log("train/brier_score", brier_score, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/aleatoric_confidence", aleatoric_confidence.mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/epistemic_confidence", epistemic_confidence.mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/calibration_metric", calibration_metric, on_step=False, on_epoch=True, prog_bar=False)
        
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "y_prob_0":y_prob[:,0],"y_prob_1":y_prob[:,1],"y_pred": y_pred, "y_pred_latent": y_pred_latent , "targets": targets, log_prob: log_prob}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        auc = self.train_auroc.compute()
        self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=False)        
        self.train_acc.reset()
        self.train_auroc.reset()


    def validation_step(self, batch: Any, batch_idx: int):
        loss, y_prob, y_pred, y_pred_latent, targets, log_prob = self.step(batch)
        self.val_auroc.update(y_prob[:,1],targets)

        # log train metrics
        acc = self.val_acc(y_pred, targets)
        brier_score = self.brier_score(y_pred_latent, targets)
        aleatoric_confidence = self.net.aleatoric_confidence(batch["image"])
        epistemic_confidence = self.net.epistemic_confidence(batch["image"])

        # log calibration metrics
        # We try to maximize the prediction confidence (epistemic_confidence) of the model while minimizing the calibration error (brier_score).
        # Hence we maximize the calibration metric.
        #calibration_metric = 0.33*epistemic_confidence.mean() - 0.33*brier_score + 0.33*aleatoric_confidence.mean()

        # log best so far validation accuracy
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/brier_score", brier_score, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/aleatoric_confidence", aleatoric_confidence.mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/epistemic_confidence", epistemic_confidence.mean(), on_step=False, on_epoch=True, prog_bar=False)
        #self.log("val/calibration_metric", calibration_metric, on_step=False, on_epoch=True, prog_bar=False)
        
        #     random_image_id = np.random.randint(0,len(batch[0]))
        #     image = batch[0][random_image_id]
        #     attention_map = self.net.visualize_attention(batch[0], grid=True)[random_image_id]
            
        #     self.logger.experiment.log({"Attention map": plt.imshow(attention_map.T)})
        #     self.logger.experiment.log({"Input": plt.imshow(image.T)})


        return {"loss": loss, "y_prob_0":y_prob[:,0],"y_prob_1":y_prob[:,1],"y_pred": y_pred, "y_pred_latent": y_pred_latent , "targets": targets, "log_prob": log_prob, "epistemic_confidence": epistemic_confidence, "aleatoric_confidence": aleatoric_confidence}
    
    def confusion_matrix(self, preds, targets, class_names = None):
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        confusion_matrix = wandb.plot.confusion_matrix(preds=preds, y_true=targets, class_names=class_names)

        return confusion_matrix

    def validation_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([batch["y_pred"] for batch in outputs])
        targets = torch.cat([batch["targets"].flatten() for batch in outputs])

        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()
        auc = self.val_auroc.compute()
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=False)      
        self.val_auroc.reset()

        if not hasattr(self,"val_ood_dataset"):
            id_dataset = self.trainer.datamodule.val_dataloader().dataset
            ood_dataset = self.ood_datamodule.val_dataloader().dataset
            min_len = min(len(id_dataset), len(ood_dataset))
            id_dataset = torch.utils.data.Subset(id_dataset, range(min_len))
            ood_dataset = torch.utils.data.Subset(ood_dataset, range(min_len))
            self.val_ood_dataset = OodDataset(id_dataset, ood_dataset)

        # log train metrics
        if self.current_epoch >= 0:
            auc_results = \
                self.compute_auc_roc(self.val_ood_dataset,output_tsne=self.hparams.output_val_tsne,output_data=self.hparams.output_val_tsne)
            self.log("val/aleatoric_AUPRC", auc_results['aleatoric_confidence_auc_pr'], on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/aleatoric_AUROC", auc_results['aleatoric_confidence_auc_roc'], on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/epistemic_AUPRC", auc_results['epistemic_confidence_auc_pr'], on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/epistemic_AUROC", auc_results['epistemic_confidence_auc_roc'], on_step=False, on_epoch=True, prog_bar=False)

            self.log("val/aleatoric_AUPRC_MacroAvg", np.mean(auc_results['group_al_auprc']), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/aleatoric_AUROC_MacroAvg", np.mean(auc_results['group_al_auroc']), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/epistemic_AUPRC_MacroAvg", np.mean(auc_results['group_ep_auprc']), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/epistemic_AUROC_MacroAvg", np.mean(auc_results['group_ep_auroc']), on_step=False, on_epoch=True, prog_bar=False)
            
            for group in [0,1]:
                self.log(f"val/aleatoric_AUPRC_{group}", auc_results['group_al_auprc'][group], on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"val/aleatoric_AUROC_{group}", auc_results['group_al_auroc'][group], on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"val/epistemic_AUPRC_{group}", auc_results['group_ep_auprc'][group], on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"val/epistemic_AUROC_{group}", auc_results['group_ep_auroc'][group], on_step=False, on_epoch=True, prog_bar=False)

            


        


            # confusion_matrix = self.confusion_matrix(preds, targets, class_names = list(self.trainer.datamodule.data_train.reduced_labels.values()))
            # self.logger.experiment.log({"confusion_matrix": confusion_matrix})

            # log calibration metrics
            calibration_metric = (auc_results['aleatoric_confidence_auc_pr'] + \
                                  auc_results['aleatoric_confidence_auc_roc'] + \
                                  auc_results['epistemic_confidence_auc_pr'] + \
                                  auc_results['epistemic_confidence_auc_roc'])/8 + acc/4
            self.log("val/calibration_metric", calibration_metric, on_step=False, on_epoch=True, prog_bar=False)

    
    def test_step(self, batch: Any, batch_idx: int):
        loss, y_prob, y_pred, y_pred_latent, targets, log_prob = self.step(batch)
        self.test_auroc.update(y_prob[:,1],targets)

        # log train metrics
        acc = self.test_acc(y_pred, targets)
        brier_score = self.brier_score(y_pred_latent, targets)
        aleatoric_confidence = self.net.aleatoric_confidence(batch["image"])
        epistemic_confidence = self.net.epistemic_confidence(batch["image"])

        # log calibration metrics
        # We try to maximize the prediction confidence of the model while minimizing the calibration error.
        #calibration_metric = 0.33*epistemic_confidence.mean() - 0.33*brier_score + 0.33*aleatoric_confidence.mean()

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/brier_score", brier_score, on_step=False, on_epoch=True)
        self.log("test/aleatoric_confidence", aleatoric_confidence.mean(), on_step=False, on_epoch=True)
        self.log("test/epistemic_confidence", epistemic_confidence.mean(), on_step=False, on_epoch=True)
        #self.log("test/calibration_metric", calibration_metric, on_step=False, on_epoch=True)

        # # Aleatoric confidence (from negative uncertainty)
        # aleatoric_conf = -preds.maximum_a_posteriori().uncertainty()
        # if aleatoric_conf.dim() > 1:
        #     aleatoric_conf = aleatoric_conf.mean(tuple(range(1, aleatoric_conf.dim())))

        # self.alea_conf_pr.update(aleatoric_conf, targets)
        # self.log(f"test/aleatoric_confidence_auc_pr", self.alea_conf_pr)

        # self.alea_conf_roc.update(aleatoric_conf, targets)
        # self.log(f"test/aleatoric_confidence_auc_roc", self.alea_conf_roc)

        # # Epistemic confidence
        # epistemic_conf = log_prob
        # if epistemic_conf.dim() > 1:
        #     epistemic_conf = epistemic_conf.mean(tuple(range(1, epistemic_conf.dim())))

        # self.epist_conf_pr.update(epistemic_conf, targets)
        # self.log(f"test/epistemic_confidence_auc_pr", self.epist_conf_pr)

        # self.epist_conf_roc.update(epistemic_conf, targets)
        # self.log(f"test/epistemic_confidence_auc_roc", self.epist_conf_roc)

        return {"loss": loss, "y_prob_0":y_prob[:,0],"y_prob_1":y_prob[:,1],"y_pred": y_pred, "y_pred_latent": y_pred_latent , "targets": targets, "log_prob": log_prob, "epistemic_confidence": epistemic_confidence, "aleatoric_confidence": aleatoric_confidence}

    def test_epoch_end(self, outputs: List[Any]):
        epistemic_confidence = torch.cat([batch["epistemic_confidence"] for batch in outputs])
        aleatoric_confidence = torch.cat([batch["aleatoric_confidence"] for batch in outputs])


        if not hasattr(self,"test_ood_dataset"):
            id_dataset = self.trainer.datamodule.test_dataloader().dataset
            # combine the training and testing subsets
            ood_test_dataset = self.ood_datamodule.test_dataloader().dataset
            ood_train_dataset = self.ood_datamodule.train_dataloader().dataset
            ood_dataset = ConcatDataset([ood_train_dataset,ood_test_dataset])


            # min_len = min(len(id_dataset), len(ood_dataset))
            # id_dataset = torch.utils.data.Subset(id_dataset, range(min_len))
            # ood_dataset = torch.utils.data.Subset(ood_dataset, range(min_len))
            self.test_ood_dataset = OodDataset(id_dataset, ood_dataset)


        
        auc_results = \
            self.compute_auc_roc(self.test_ood_dataset,output_tsne=True,output_data=True)
            # self.compute_auc_roc(self.val_ood_dataset,output_tsne=self.hparams.output_val_tsne,output_data=self.hparams.output_val_tsne)

        self.log("test/aleatoric_AUPRC", auc_results['aleatoric_confidence_auc_pr'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/aleatoric_AUROC", auc_results['aleatoric_confidence_auc_roc'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/epistemic_AUPRC", auc_results['epistemic_confidence_auc_pr'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/epistemic_AUROC", auc_results['epistemic_confidence_auc_roc'], on_step=False, on_epoch=True, prog_bar=False)

        self.log("test/aleatoric_AUPRC_MacroAvg", np.mean(auc_results['group_al_auprc']), on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/aleatoric_AUROC_MacroAvg", np.mean(auc_results['group_al_auroc']), on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/epistemic_AUPRC_MacroAvg", np.mean(auc_results['group_ep_auprc']), on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/epistemic_AUROC_MacroAvg", np.mean(auc_results['group_ep_auroc']), on_step=False, on_epoch=True, prog_bar=False)
        
        for group in [0,1]:
            self.log(f"test/aleatoric_AUPRC_{group}", auc_results['group_al_auprc'][group], on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"test/aleatoric_AUROC_{group}", auc_results['group_al_auroc'][group], on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"test/epistemic_AUPRC_{group}", auc_results['group_ep_auprc'][group], on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"test/epistemic_AUROC_{group}", auc_results['group_ep_auroc'][group], on_step=False, on_epoch=True, prog_bar=False)

        # confusion_matrix = self.confusion_matrix(preds, targets, class_names = list(self.trainer.datamodule.data_train.reduced_labels.values()))
        # self.logger.experiment.log({"confusion_matrix": confusion_matrix})

        # log calibration metrics
        calibration_metric = (auc_results['aleatoric_confidence_auc_pr'] + \
                                auc_results['aleatoric_confidence_auc_roc'] + \
                                auc_results['epistemic_confidence_auc_pr'] + \
                                auc_results['epistemic_confidence_auc_roc'])/4
        self.log("test/calibration_metric", calibration_metric, on_step=False, on_epoch=True, prog_bar=False)

        # # log calibration metrics
        # (aleatoric_confidence_auc_pr, aleatoric_confidence_auc_roc, epistemic_confidence_auc_pr, epistemic_confidence_auc_roc) = \
        #     self.compute_auc_roc(self.test_ood_dataset,output_tsne=True,output_data=True)
        # self.log("test/aleatoric_confidence_auc_pr", aleatoric_confidence_auc_pr, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("test/aleatoric_confidence_auc_roc", aleatoric_confidence_auc_roc, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("test/epistemic_confidence_auc_pr", epistemic_confidence_auc_pr, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("test/epistemic_confidence_auc_roc", epistemic_confidence_auc_roc, on_step=False, on_epoch=True, prog_bar=False)

        # calibration_metric = (aleatoric_confidence_auc_pr + aleatoric_confidence_auc_roc + epistemic_confidence_auc_pr + epistemic_confidence_auc_roc)/4
        # self.log("test/calibration_metric", calibration_metric, on_step=False, on_epoch=True, prog_bar=True)

        self.test_acc.reset()

        auc = self.test_auroc.compute()
        self.log("test/auc", auc, on_step=False, on_epoch=True, prog_bar=False)      
        self.test_auroc.reset()


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters(),lr=self.hparams.lr),
            
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                self.hparams.optimizer(params=self.parameters()),
                eta_min = 1e-6, T_max=len(self.trainer._data_connector._train_dataloader_source.dataloader())),
            "monitor": "metric_to_track",
            "frequency": "indicates how often the metric is updated"
        }
    
    def compute_epistemic_aleatoric(self, ood_dataset, limit = 300,output_tsne=False,output_data=True):
        """
        Compute the AUC-ROC values for aleatoric and epistemic confidence on the given out-of-distribution dataset using
        the given model.

        Args:
            ood_dataset: An instance of the out-of-distribution dataset to use.
            model: The model to use for computing the confidence scores.
            limit: The maximum number of batches to use from the out-of-distribution dataset.

        Returns:
            A tuple of the AUC-ROC values for aleatoric and epistemic confidence.
        """
        # set torch random seed
        dataloader = torch.utils.data.DataLoader(ood_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)

        # Compute all the y_true and y_score_epistemic over the first 100 batches
        y_true = []
        ID_label = []
        ID_prob = []

        y_score_epistemic = []
        y_score_aleatoric = []
        patient_id = []
        img_paths = []
        embeddings = []
        log_entropies = []

        # set numpy random seed

        with torch.no_grad():
            for batch in tqdm(dataloader):
                y_true.append(batch["id_data"].detach().numpy())
                ID_label.append(batch["label"].detach().numpy())
                loss, y_prob, y_pred, y_pred_latent, targets, log_prob = self.step(batch)
                ID_prob.append(y_prob)
            
                y_score_epistemic.append(self.net.epistemic_confidence(batch["image"].to(self.hparams.device)).detach().cpu().numpy())
                y_score_aleatoric.append(self.net.aleatoric_confidence(batch["image"].to(self.hparams.device)).detach().cpu().numpy())
                patient_id.append(batch["patient_id"])
                img_paths.append(batch["path"])
                embeddings.append(self.net.encoder(batch["image"].to(self.hparams.device)).detach().cpu().numpy())
                log_entropies.append(-self.net(batch["image"].to(self.hparams.device))[0].maximum_a_posteriori().uncertainty().detach().cpu().numpy())

        # Concatenate the y_true and y_score_epistemic lists
        y_true = np.concatenate(y_true)
        patient_id = np.concatenate(patient_id)
        img_paths = np.concatenate(img_paths)
        y_score_epistemic = np.concatenate(y_score_epistemic)
        y_score_aleatoric = np.concatenate(y_score_aleatoric)
        embeddings = np.concatenate(embeddings)
        ID_label = np.concatenate(ID_label)
        ID_prob = np.concatenate(ID_prob)

        # Export a csv with a counter, patient_id, y_score_epistemic and y_true
        # scale the aleatoric confidence to be between 0 and 1

        #y_score_aleatoric = (y_score_aleatoric - y_score_aleatoric.min())/(y_score_aleatoric.max() - y_score_aleatoric.min())
        if output_tsne:
            self.export_tsne_figure(y_true, embeddings, seed=self.hparams.seed, log_entropy=y_score_epistemic)
        
        df = pd.DataFrame({
            'patient_id': patient_id,
            'img_path': img_paths,
            'y_score_epistemic': y_score_epistemic,
            'y_score_aleatoric': y_score_aleatoric,
            'prob1': y_score_epistemic,
            'ID_label': ID_label,
            'ID_prob':ID_prob[:,1],
            'label': y_true})

        # shuffle dtaframe
        # df = df.sample(frac=1).reset_index(drop=True)
        
        DATA_OUTPATH = os.path.join(self.trainer.logger._save_dir,'data',f'epoch_{self.current_epoch}')
        REPORT_OUTPATH = os.path.join(self.trainer.logger._save_dir,'report',f'epoch_{self.current_epoch}','figures')
        os.makedirs(DATA_OUTPATH,exist_ok=True)
        os.makedirs(REPORT_OUTPATH,exist_ok=True)
        if output_data:
            df.to_csv(os.path.join(DATA_OUTPATH,f'postnet_{self.hparams.seed}.csv'), index=True)
            # export embeddings to numpy format
            np.save(os.path.join(DATA_OUTPATH,f'postnet_{self.hparams.seed}.npy'), embeddings)

        # Save a ROC curve for the epistemic confidence

        fpr, tpr, thresholds = roc_curve(y_true, y_score_epistemic)
        roc_auc = auc(fpr, tpr)

        plt.figure()

        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        plt.xlim([0.0, 1.0])

        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.title('ROC curve')

        plt.legend(loc="lower right")

        # save the image
        
        plt.savefig(os.path.join(REPORT_OUTPATH,f'roc_curve_epistemic_{self.hparams.seed}.pdf'))

#        self.logger.experiment.log({"roc_curve_epistemic": wandb.Image(plt)})

        # return y_true, y_score_epistemic, y_score_aleatoric
        return df

    def export_tsne_figure(self, y_true, embeddings, log_entropy, seed=42):

        # Visualize the t-sne
        tsne = TSNE(n_components=2, init="pca", random_state=seed, n_jobs=6, learning_rate="auto", perplexity=7)
        embeddings_tsne = tsne.fit_transform(embeddings)

        DATA_OUTPATH = os.path.join(self.trainer.logger._save_dir,'data',f'epoch_{self.current_epoch}')
        REPORT_OUTPATH = os.path.join(self.trainer.logger._save_dir,'report',f'epoch_{self.current_epoch}','figures')
        os.makedirs(DATA_OUTPATH,exist_ok=True)
        os.makedirs(REPORT_OUTPATH,exist_ok=True)

        # Create a color map for the two classes using the viridis colormap
        colors = {0: (0.267004, 0.004874, 0.329415, 1.0), 1: (0.993248, 0.906157, 0.143936, 1.0)}
        labels = {0: 'In-Distribution', 1: 'Out-of-Distribution'}
        class_markers =['^' if t else 'o' for t in y_true]
        class_labels = np.array([labels[t] for t in y_true])
        class_colors = np.array([colors[t] for t in y_true])


        # Compute F1 score for each threshold value
        thresholds = sorted(log_entropy)
        thresholds = thresholds[2:-2]
        f1_scores = []
        for threshold in thresholds:
            y_pred = (log_entropy >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1_score)

        # Find the threshold that maximizes the F1 score
        max_f1_score = max(f1_scores)
        optimal_threshold = thresholds[f1_scores.index(max_f1_score)]

        # Set up the figure with two subplots
        fig1, ax1 = plt.subplots(figsize=(13,10))
        fig2, ax2 = plt.subplots(figsize=(15,10))
        fig3, ax3 = plt.subplots(figsize=(15,10))

        # Plot the t-SNE embeddings with class labels
        for label in np.unique(class_labels):
            idxs = np.where(class_labels == label)
            ax1.scatter(embeddings_tsne[idxs, 0], embeddings_tsne[idxs, 1], c=class_colors[idxs], label=label, alpha=0.5, s=4, cmap="RdYlGn")

        # Set plot title and labels
        ax1.set_title("Latent Space Embedding")
        ax1.set_xlabel("t-SNE Component 1")
        ax1.set_ylabel("t-SNE Component 2")

        # Add legend
        ax1.legend(loc="upper left")

        fig1.savefig(os.path.join(REPORT_OUTPATH,f"latent_space_{seed}.pdf"), dpi=600)
        from sklearn.preprocessing import MinMaxScaler

        # create scaler object
        scaler = MinMaxScaler(feature_range=(0, 1))

        # assume data is stored in a numpy array called 'data'
        # create a mask that identifies elements above and below the threshold
        mask = log_entropy >= optimal_threshold

        # create a copy of the data array to work with
        scaled_data = np.copy(log_entropy)

        # apply different scaling factors to the elements above and below the threshold
        if len(scaled_data[~mask])>0:
            scaled_data[~mask] = scaler.fit_transform(scaled_data[~mask].reshape(-1, 1)).ravel() * 0.5
        if len(scaled_data[mask])>0:
            scaled_data[mask] = scaler.fit_transform(scaled_data[mask].reshape(-1, 1)).ravel() * 0.5 + 0.5
        

        # the resulting 'scaled_data' array will have values between 0 and 1,
        # with values under the threshold mapped to values between 0 and 0.5,
        # and values above the threshold mapped to values between 0.5 and 1

        # Plot the log entropy values
        im = ax2.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=scaled_data, cmap='viridis', s=4)
        ax2.set_title("Predicted uncertainty")
        ax2.set_xlabel("t-SNE Component 1")
        ax2.set_ylabel("t-SNE Component 2")

        # Add color bar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("Predicted uncertainty")

        # Save the figures
        fig2.savefig(os.path.join(REPORT_OUTPATH,f"predicted_entropy_{seed}.pdf"), dpi=600)

        class_markers =['^' if t else 'o' for t in y_true]


        # ax3.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],marker = class_markers, c=scaled_data, cmap='viridis', s=4)
        im = ax3.scatter(embeddings_tsne[y_true==1, 0], embeddings_tsne[y_true==1, 1],marker = '^', c=scaled_data[y_true==1], cmap='viridis', s=4,label='ID')
        im2 = ax3.scatter(embeddings_tsne[y_true==0, 0], embeddings_tsne[y_true==0, 1],marker = 'o', c=scaled_data[y_true==0], cmap='viridis', s=4,label='OOD')

        # Add legend
        ax3.legend(loc="upper left")
        ax3.set_title("Predicted uncertainty")
        ax3.set_xlabel("t-SNE Component 1")
        ax3.set_ylabel("t-SNE Component 2")

        # Add color bar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label("Predicted uncertainty")

        # Save the figures
        fig3.savefig(os.path.join(REPORT_OUTPATH,f"predicted_entropy_with_markers{seed}.pdf"), dpi=600)

        # export the embeddings along with the labels and log entropy to a csv
        df = pd.DataFrame({'x': embeddings_tsne[:, 0], 'y': embeddings_tsne[:, 1], 'label': y_true, 'log_entropy': scaled_data})
        df.to_csv(os.path.join(DATA_OUTPATH,f'tsne_{self.hparams.seed}.csv'), index=True)
        



    def compute_auc_roc(self, ood_dataset,output_tsne=False,output_data=True):
        """
        Compute the AUC-ROC values for aleatoric and epistemic confidence on the given out-of-distribution dataset using
        the given model.

        Args:
            ood_dataset: An instance of the out-of-distribution dataset to use.
            model: The model to use for computing the confidence scores.

        Returns:
            A tuple containing the AUC-ROC values for aleatoric confidence AUC-PR, aleatoric confidence AUC-ROC,
            epistemic confidence AUC-PR, and epistemic confidence AUC-ROC, respectively.
        """
        # y_true, y_score_epistemic, y_score_aleatoric = self.compute_epistemic_aleatoric(ood_dataset,output_tsne=output_tsne,output_data=output_data)
        df = self.compute_epistemic_aleatoric(ood_dataset,output_tsne=output_tsne,output_data=output_data)
        y_true = df['label'].values
        y_score_epistemic = df['y_score_epistemic'].values
        y_score_aleatoric = df['y_score_aleatoric'].values

        # Compute AUC-PR and AUC-ROC for aleatoric and epistemic confidence
        aleatoric_confidence_auc_pr = AUCPR(compute_on_step=False)( torch.tensor(y_score_aleatoric),torch.tensor(y_true))
        aleatoric_confidence_auc_roc = AUROC(compute_on_step=False)( torch.tensor(y_score_aleatoric),torch.tensor(y_true))
        epistemic_confidence_auc_pr = AUCPR(compute_on_step=False)( torch.tensor(y_score_epistemic),torch.tensor(y_true))
        epistemic_confidence_auc_roc = AUROC(compute_on_step=False)( torch.tensor(y_score_epistemic),torch.tensor(y_true))
        # get groupwise AUC-PR and AUC-ROC
        group_al_auroc = np.zeros((2,)) # aleatoric confidence AUC-ROC
        group_ep_auroc = np.zeros((2,)) # epistemic confidence AUC-ROC
        group_al_auprc = np.zeros((2,)) # aleatoric confidence AUC-PR
        group_ep_auprc = np.zeros((2,)) # epistemic confidence AUC-PR
        for group in [0,1]:
            # keep only 1 in-distribution group and the out-of-distribution group
            df_group = df.loc[(df['ID_label']==group) | (df['label']==0)].reset_index(drop=True)
            y_true = df_group['label'].values
            y_score_epistemic = df_group['y_score_epistemic'].values
            y_score_aleatoric = df_group['y_score_aleatoric'].values
            # Compute AUC-PR and AUC-ROC for aleatoric and epistemic confidence
            group_al_auroc[group] = AUROC(compute_on_step=False)( torch.tensor(y_score_aleatoric),torch.tensor(y_true))
            group_ep_auroc[group] = AUROC(compute_on_step=False)( torch.tensor(y_score_epistemic),torch.tensor(y_true))
            group_al_auprc[group] = AUCPR(compute_on_step=False)( torch.tensor(y_score_aleatoric),torch.tensor(y_true))
            group_ep_auprc[group] = AUCPR(compute_on_step=False)( torch.tensor(y_score_epistemic),torch.tensor(y_true))
        
        results = {
            'aleatoric_confidence_auc_pr': aleatoric_confidence_auc_pr,
            'aleatoric_confidence_auc_roc': aleatoric_confidence_auc_roc,
            'epistemic_confidence_auc_pr': epistemic_confidence_auc_pr,
            'epistemic_confidence_auc_roc': epistemic_confidence_auc_roc,
            'group_al_auroc': group_al_auroc,
            'group_ep_auroc': group_ep_auroc,
            'group_al_auprc': group_al_auprc,
            'group_ep_auprc': group_ep_auprc
        }
        return results

        # return (aleatoric_confidence_auc_pr, aleatoric_confidence_auc_roc, epistemic_confidence_auc_pr, epistemic_confidence_auc_roc)

class UncertaintyModuleCTrans(UncertaintyModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer = None,
        exclude_uncertain: bool = False,
        net=loadCTransEncoder,
        latent_dim: int = 5,
        clamp_scaling: bool = True,
        radial_flow_layers: int = 6,
        flow_type: Literal["radial","MAF"] = "radial",
        certainty_budget: CertaintyBudget = "normal",
        entropy_weight: float = 1e-5,
        number_of_classes: int = 5,
        binary_label: int = None,
        locked_encoder: bool = False,
        locked_classifier: bool=False,
        cancer_ood: bool=True,
        extra_ood: bool=False,
        seed: int = 0,
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=1000,
        output_val_tsne=False,
        lr=0.001
    ):
        super(UncertaintyModule, self).__init__() 
        CTRANS_LATENT_DIM=768
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        ood_datamodule = EbrainDataModule(batch_size=batch_size, num_workers=4, cancer_ood=cancer_ood, extra_ood=extra_ood, exclude_uncertain=exclude_uncertain)
        ood_datamodule.setup()
        self.ood_datamodule = ood_datamodule

        self.save_hyperparameters(logger=False, ignore=["net"])
        print("hparams:")
        print(self.hparams)
        encoder = net()
        if self.hparams["locked_encoder"] == True:
            for param in encoder.parameters():
                param.requires_grad = False
        if self.hparams["latent_dim"] is not None:
            encoder_2 = MLP(
                CTRANS_LATENT_DIM, fc_latent_size=[], num_classes=self.hparams["latent_dim"], dropout=0.0)
        else:
            encoder_2 = nn.Identity()
            self.hparams["latent_dim"] = CTRANS_LATENT_DIM
        encoder = nn.Sequential(encoder,encoder_2)
        if locked_classifier:
            classifier = loadCTransClassifier()
            for param in classifier.linear.parameters():
                param.requires_grad = False
        else:
            classifier=CategoricalOutput(
                self.hparams["latent_dim"], self.hparams["number_of_classes"]
            )
        flow_class = FLOW_TYPE_DICT[self.hparams['flow_type']]

        self.net = NaturalPosteriorNetworkModel(
            latent_dim=self.hparams["latent_dim"],
            encoder=encoder,
            flow = flow_class(self.hparams["latent_dim"], num_layers=self.hparams["radial_flow_layers"]),
            certainty_budget=self.hparams["certainty_budget"],
            output=classifier,
            clamp_scaling = self.hparams["clamp_scaling"],
        )

        # loss function
        self.criterion = BayesianLoss(entropy_weight=self.hparams["entropy_weight"])

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # task_type='binary'
        # self.train_acc = Accuracy(multidim_average="global")
        # self.val_acc = Accuracy(multidim_average="global")
        # self.test_acc = Accuracy(multidim_average="global")
        self.train_acc = Accuracy(multidim_average="global")
        self.val_acc = Accuracy(multidim_average="global")
        self.test_acc = Accuracy(multidim_average="global")

        # self.train_auroc = AUROC(task='binary')
        # self.val_auroc = AUROC(task='binary')
        # self.test_auroc = AUROC(task='binary')
        self.train_auroc = AUROC()
        self.val_auroc = AUROC()
        self.test_auroc = AUROC()
        # We have discrete output
        self.output = "discrete"
        self.brier_score = BrierScore(
            compute_on_step=False, dist_sync_fn=self.all_gather, full_state_update=False
        )

        self.alea_conf_pr = AUCPR(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.alea_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

        self.epist_conf_pr = AUCPR(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.epist_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # self.ood_datamodule = None

class UncertaintyModuleCTransFeatures(UncertaintyModule):
    # Use encoded CTransPath features instead of images
    def __init__(
        self,
        optimizer: torch.optim.Optimizer = None,
        exclude_uncertain: bool = False,
        clamp_scaling: bool = True,
        net=nn.Identity,
        latent_dim: int = 5,
        radial_flow_layers: int = 6,
        flow_type: Literal["radial","MAF"] = "radial",
        certainty_budget: CertaintyBudget = "normal",
        entropy_weight: float = 1e-5,
        number_of_classes: int = 5,
        binary_label: int = None,
        locked_encoder: bool = False,
        locked_classifier: bool=False,
        cancer_ood: bool=True,
        extra_ood: bool=False,
        seed: int = 0,
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=1000,
        output_val_tsne=False,
        lr=0.001
    ):
        super(UncertaintyModule, self).__init__() 
        CTRANS_LATENT_DIM=768
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        ood_datamodule = EbrainFeatureDataModule(
            batch_size=batch_size,num_workers=4, cancer_ood=cancer_ood, extra_ood=extra_ood, exclude_uncertain=exclude_uncertain)
        ood_datamodule.setup()
        self.ood_datamodule = ood_datamodule

        self.save_hyperparameters(logger=False, ignore=["net"])
        print("hparams:")
        print(self.hparams)
        encoder = net()
        if self.hparams["locked_encoder"] == True:
            for param in encoder.parameters():
                param.requires_grad = False
        if self.hparams["latent_dim"] is not None:
            encoder_2 = MLP(
                CTRANS_LATENT_DIM, fc_latent_size=[], num_classes=self.hparams["latent_dim"], dropout=0.0)
        else:
            encoder_2 = nn.Identity()
            self.hparams["latent_dim"] = CTRANS_LATENT_DIM
        encoder = nn.Sequential(encoder,encoder_2)
        if locked_classifier:
            classifier = loadCTransClassifier()
            for param in classifier.linear.parameters():
                param.requires_grad = False
        else:
            classifier=CategoricalOutput(
                self.hparams["latent_dim"], self.hparams["number_of_classes"]
            )
        flow_class = FLOW_TYPE_DICT[self.hparams['flow_type']]

        self.net = NaturalPosteriorNetworkModel(
            latent_dim=self.hparams["latent_dim"],
            encoder=encoder,
            flow = flow_class(self.hparams["latent_dim"], num_layers=self.hparams["radial_flow_layers"]),
            certainty_budget=self.hparams["certainty_budget"],
            clamp_scaling = self.hparams["clamp_scaling"],
            output=classifier
        )

        # loss function
        self.criterion = BayesianLoss(entropy_weight=self.hparams["entropy_weight"])

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # task_type='binary'
        # self.train_acc = Accuracy(multidim_average="global")
        # self.val_acc = Accuracy(multidim_average="global")
        # self.test_acc = Accuracy(multidim_average="global")
        self.train_acc = Accuracy(multidim_average="global")
        self.val_acc = Accuracy(multidim_average="global")
        self.test_acc = Accuracy(multidim_average="global")

        # self.train_auroc = AUROC(task='binary')
        # self.val_auroc = AUROC(task='binary')
        # self.test_auroc = AUROC(task='binary')
        self.train_auroc = AUROC()
        self.val_auroc = AUROC()
        self.test_auroc = AUROC()
        # We have discrete output
        self.output = "discrete"
        self.brier_score = BrierScore(
            compute_on_step=False, dist_sync_fn=self.all_gather, full_state_update=False
        )

        self.alea_conf_pr = AUCPR(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.alea_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

        self.epist_conf_pr = AUCPR(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.epist_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # self.ood_datamodule = None

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "slides.yaml")
    _ = hydra.utils.instantiate(cfg)

# Segformer model for PyTorch using Huggingface Transformer library