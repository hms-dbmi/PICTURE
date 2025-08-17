from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, AUROC
from torchmetrics.classification.accuracy import Accuracy

from src.models.slides_module import ViT8

from typing import Any, List
import torch
from src.models.natpn.metrics import BrierScore, AUCPR
from src.models.natpn.nn import BayesianLoss, NaturalPosteriorNetworkModel
from src.models.natpn.nn.output import CategoricalOutput
from src.models.natpn.nn.flow.radial import RadialFlow
from src.models.natpn.nn.model import NaturalPosteriorNetworkModel
import wandb

from tqdm import tqdm
from src.datamodules.cns_datamodule import CNSDataModule
from src.datasets.ood_dataset import OodDataset

import torch.nn as nn
import torch.nn.functional as F

class DropoutModule(LightningModule):
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
        number_of_classes: int = 5,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = DropoutClassifier(number_of_classes=self.hparams["number_of_classes"])
        
        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy(mdmc_reduce="global")
        self.val_acc = Accuracy(mdmc_reduce="global")
        self.test_acc = Accuracy(mdmc_reduce="global")

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
        y_pred = self.net(x)
        y_pred_labels = y_pred.argmax(1)
        loss = self.criterion(y_pred, y)

        return loss, y_pred, y, y_pred_labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, y_pred, targets, y_pred_labels = self.step(batch)

        # log train metrics
        acc = self.train_acc(y_pred_labels, targets)


        # log calibration metrics
        # We try to maximize the prediction confidence (epistemic_confidence) of the model while minimizing the calibration error (brier_score).
        # Hence we maximize the calibration metric.
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)        

        
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "y_pred": y_pred, "targets": targets, "y_pred_labels": y_pred_labels}
    
    def uncertainty_step(self, batch: Any):
        x, y = batch["image"], batch["label"]
        uncertainty = self.net.uncertainty(x)
        return uncertainty.mean()

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, y_pred, targets, y_pred_labels = self.step(batch)
        uncertainty = self.uncertainty_step(batch)

        # log train metrics
        acc = self.val_acc(y_pred_labels, targets)

        # log best so far validation accuracy
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc_best.update(self.val_acc.compute())
        self.log("val/uncertainty", uncertainty, on_step=False, on_epoch=True, prog_bar=False)

        #self.log("val/calibration_metric", calibration_metric, on_step=False, on_epoch=True, prog_bar=False)
        
        #     random_image_id = np.random.randint(0,len(batch[0]))
        #     image = batch[0][random_image_id]
        #     attention_map = self.net.visualize_attention(batch[0], grid=True)[random_image_id]
            
        #     self.logger.experiment.log({"Attention map": plt.imshow(attention_map.T)})
        #     self.logger.experiment.log({"Input": plt.imshow(image.T)})


        return {"loss": loss, "y_pred": y_pred, "targets": targets, "y_pred_labels": y_pred_labels, "uncertainty": uncertainty}
    

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, y_pred, targets, y_pred_labels = self.step(batch)
        uncertainty = self.uncertainty_step(batch)

        # log train metrics
        acc = self.val_acc(y_pred_labels, targets)

        # log best so far validation accuracy
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/uncertainty", uncertainty, on_step=False, on_epoch=True, prog_bar=False)


        return {"loss": loss, "y_pred": y_pred, "targets": targets, "y_pred_labels": y_pred_labels, "uncertainty": uncertainty}
    
    def test_epoch_end(self, outputs: List[Any]):
        self.test_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(self.hparams.optimizer(params=self.parameters()), eta_min = 1e-6, T_max=len(self.trainer._data_connector._train_dataloader_source.dataloader())),
            "monitor": "metric_to_track",
            "frequency": "indicates how often the metric is updated"
        }


class DropoutClassifier(nn.Module):
    def __init__(self, number_of_classes, dropout_rate=0.5):
        super(DropoutClassifier, self).__init__()
        self.number_of_classes = number_of_classes
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, number_of_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict(self, x):
        self.eval()  # set model to evaluation mode
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            return probs

    def uncertainty(self, x, num_samples=25):
        self.train()  # set model to training mode
        with torch.no_grad():
            logits_list = []
            for _ in range(num_samples):
                logits = self(x)
                logits_list.append(logits.unsqueeze(0))
            logits_tensor = torch.cat(logits_list, dim=0)
            probs_tensor = F.softmax(logits_tensor, dim=2)
            mean_probs = probs_tensor.mean(dim=0)
            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1)
            return entropy

    def detect_ood(self, x, threshold):
        probs = self.predict(x)
        entropy = self.uncertainty(x)
        is_ood = entropy > threshold
        return probs, entropy, is_ood



if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "slides.yaml")
    _ = hydra.utils.instantiate(cfg)

# Segformer model for PyTorch using Huggingface Transformer library


