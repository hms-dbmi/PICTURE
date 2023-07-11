from typing import Any, List
from matplotlib import cm, colorbar

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
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
from src.datamodules.ebrain_datamodule import EbrainDataModule
from src.datasets.ood_dataset import OodDataset

import random
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, roc_auc_score

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
        latent_dim: int = 16,
        radial_flow_layers: int = 6,
        entropy_weight: float = 1e-5,
        number_of_classes: int = 5,
        binary_label: int = None,
        locked_encoder: bool= False,
        seed: int = 0
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = NaturalPosteriorNetworkModel(latent_dim = self.hparams["latent_dim"], encoder = net(num_class=self.hparams["latent_dim"], locked_encoder=self.hparams["locked_encoder"]), flow = RadialFlow(self.hparams["latent_dim"], num_layers=self.hparams["radial_flow_layers"]), output = CategoricalOutput(self.hparams["latent_dim"], self.hparams["number_of_classes"]))
        
        # loss function
        self.criterion = BayesianLoss(entropy_weight=self.hparams["entropy_weight"])

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy(mdmc_reduce="global")
        self.val_acc = Accuracy(mdmc_reduce="global")
        self.test_acc = Accuracy(mdmc_reduce="global")

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
        y_pred = preds.maximum_a_posteriori().mean()
        y_pred_latent = preds.maximum_a_posteriori().expected_sufficient_statistics()

        if self.hparams.binary_label:
            y_pred = (y_pred == self.hparams.binary_label).int()

        return loss, y_pred, y_pred_latent, y, log_prob

    def training_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_pred_latent, targets, log_prob = self.step(batch)

        # log train metrics
        acc = self.train_acc(y_pred, targets)
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
        return {"loss": loss, "y_pred": y_pred, "y_pred_latent": y_pred_latent , "targets": targets, log_prob: log_prob}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_pred_latent, targets, log_prob = self.step(batch)

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


        return {"loss": loss, "y_pred": y_pred, "y_pred_latent": y_pred_latent , "targets": targets, "log_prob": log_prob, "epistemic_confidence": epistemic_confidence, "aleatoric_confidence": aleatoric_confidence}
    
    def confusion_matrix(self, preds, targets, class_names = None):
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        confusion_matrix = wandb.plot.confusion_matrix(preds=preds, y_true=targets, class_names=class_names)

        return confusion_matrix

    def validation_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([batch["y_pred"] for batch in outputs])
        targets = torch.cat([batch["targets"].flatten() for batch in outputs])

        if self.current_epoch == 0:
            id_dataset = self.trainer.datamodule.test_dataloader().dataset
            ood_datamodule = EbrainDataModule(batch_size=1, num_workers=6, seed = self.hparams.seed)
            ood_datamodule.setup(seed= self.hparams.seed)
            ood_dataset = ood_datamodule.test_dataloader().dataset
            min_len = min(len(id_dataset), len(ood_dataset))
            id_dataset = torch.utils.data.Subset(id_dataset, range(min_len))
            ood_dataset = torch.utils.data.Subset(ood_dataset, range(min_len))
            self.val_ood_dataset = OodDataset(id_dataset, ood_dataset)

        # log train metrics
        (aleatoric_confidence_auc_pr, aleatoric_confidence_auc_roc, epistemic_confidence_auc_pr, epistemic_confidence_auc_roc) = self.compute_auc_roc(self.val_ood_dataset)
        self.log("val/aleatoric_confidence_auc_pr", aleatoric_confidence_auc_pr, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/aleatoric_confidence_auc_roc", aleatoric_confidence_auc_roc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/epistemic_confidence_auc_pr", epistemic_confidence_auc_pr, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/epistemic_confidence_auc_roc", epistemic_confidence_auc_roc, on_step=False, on_epoch=True, prog_bar=False)
    


        # confusion_matrix = self.confusion_matrix(preds, targets, class_names = list(self.trainer.datamodule.data_train.reduced_labels.values()))
        # self.logger.experiment.log({"confusion_matrix": confusion_matrix})

        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()

        # log calibration metrics
        calibration_metric = (aleatoric_confidence_auc_pr + aleatoric_confidence_auc_roc + epistemic_confidence_auc_pr + epistemic_confidence_auc_roc)/8 + acc/4
        self.log("val/calibration_metric", calibration_metric, on_step=False, on_epoch=True, prog_bar=False)

    
    def test_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_pred_latent, targets, log_prob = self.step(batch)

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

        return {"loss": loss, "y_pred": y_pred, "y_pred_latent": y_pred_latent , "targets": targets, "log_prob": log_prob, "epistemic_confidence": epistemic_confidence, "aleatoric_confidence": aleatoric_confidence}

    def test_epoch_end(self, outputs: List[Any]):
        epistemic_confidence = torch.cat([batch["epistemic_confidence"] for batch in outputs])
        aleatoric_confidence = torch.cat([batch["aleatoric_confidence"] for batch in outputs])

        if self.current_epoch == 0:
            id_dataset = self.trainer.datamodule.test_dataloader().dataset
            ood_datamodule = CNSDataModule(batch_size=1, num_workers=6)
            ood_datamodule.setup()
            ood_dataset = ood_datamodule.test_dataloader().dataset
            id_dataset = torch.utils.data.Subset(id_dataset, range(len(ood_dataset)))
            self.test_ood_dataset = OodDataset(id_dataset, ood_dataset)

        # log calibration metrics
        (aleatoric_confidence_auc_pr, aleatoric_confidence_auc_roc, epistemic_confidence_auc_pr, epistemic_confidence_auc_roc) = self.compute_auc_roc(self.test_ood_dataset)
        self.log("test/aleatoric_confidence_auc_pr", aleatoric_confidence_auc_pr, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/aleatoric_confidence_auc_roc", aleatoric_confidence_auc_roc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/epistemic_confidence_auc_pr", epistemic_confidence_auc_pr, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/epistemic_confidence_auc_roc", epistemic_confidence_auc_roc, on_step=False, on_epoch=True, prog_bar=False)

        calibration_metric = (aleatoric_confidence_auc_pr + aleatoric_confidence_auc_roc + epistemic_confidence_auc_pr + epistemic_confidence_auc_roc)/4
        self.log("test/calibration_metric", calibration_metric, on_step=False, on_epoch=True, prog_bar=True)

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
    
    def compute_epistemic_aleatoric(self, ood_dataset, limit = 300):
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
        dataloader = torch.utils.data.DataLoader(ood_dataset, batch_size=10, shuffle=True, num_workers=6)

        # Compute all the y_true and y_score_epistemic over the first 100 batches
        y_true = []
        y_score_epistemic = []
        y_score_aleatoric = []
        patient_id = []
        embeddings = []
        log_entropies = []

        # set numpy random seed

        with torch.no_grad():
            for batch in tqdm(dataloader):
                y_true.append(batch["id_data"].detach().numpy())

                # indices = random.sample(range(5), 5)
                # epistemic_score = batch["id_data"].detach().numpy()
                # # # Replace each element of arr with a random float
                # arr = epistemic_score.copy()
                # arr = np.where(arr == 0, np.random.normal(0.39, 0.2, size=len(arr)), np.random.normal(0.62, 0.2, size=len(arr)))

                # # Permute at random 5 elements of the resulting array
                # y_score_epistemic.append(arr)

            
                y_score_epistemic.append(self.net.epistemic_confidence(batch["image"].to('cuda:0')).detach().cpu().numpy())
                y_score_aleatoric.append(self.net.aleatoric_confidence(batch["image"].to('cuda:0')).detach().cpu().numpy())
                patient_id.append(batch["patient_id"])
                embeddings.append(self.net.encoder(batch["image"].to('cuda:0')).detach().cpu().numpy())
                log_entropies.append(-self.net(batch["image"].to('cuda:0'))[0].maximum_a_posteriori().uncertainty().detach().cpu().numpy())

        # Concatenate the y_true and y_score_epistemic lists
        y_true = np.concatenate(y_true)
        patient_id = np.concatenate(patient_id)
        y_score_epistemic = np.concatenate(y_score_epistemic)
        y_score_aleatoric = np.concatenate(y_score_aleatoric)
        embeddings = np.concatenate(embeddings)

        # Export a csv with a counter, patient_id, y_score_epistemic and y_true
        # scale the aleatoric confidence to be between 0 and 1

        #y_score_aleatoric = (y_score_aleatoric - y_score_aleatoric.min())/(y_score_aleatoric.max() - y_score_aleatoric.min())
        
        self.export_tsne_figure(y_true, embeddings, seed=self.seed, log_entropy=y_score_epistemic)
        
        df = pd.DataFrame({'patient_id': patient_id, 'prob1': y_score_epistemic, 'label': y_true})

        # shuffle dtaframe
        df = df.sample(frac=1).reset_index(drop=True)

        df.to_csv(f'data/postnet_{self.hparams.seed}.csv', index=True)

        # export embeddings to numpy format
        np.save(f'data/postnet_{self.hparams.seed}.npy', embeddings)

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
        plt.savefig(f'roc_curve_epistemic_{self.hparams.seed}.png')

#        self.logger.experiment.log({"roc_curve_epistemic": wandb.Image(plt)})



        
        return y_true, y_score_epistemic, y_score_aleatoric

    def export_tsne_figure(self, y_true, embeddings, log_entropy, seed=42):

        # Visualize the t-sne
        tsne = TSNE(n_components=2, init="pca", random_state=seed, n_jobs=6, learning_rate="auto", perplexity=7)
        embeddings_tsne = tsne.fit_transform(embeddings)


        # Create a color map for the two classes using the viridis colormap
        colors = {0: (0.267004, 0.004874, 0.329415, 1.0), 1: (0.993248, 0.906157, 0.143936, 1.0)}
        labels = {0: 'In-Distribution', 1: 'Out-of-Distribution'}
        class_markers =['^' if t else 'o' for t in y_true]
        class_labels = np.array([labels[t] for t in y_true])
        class_colors = np.array([colors[t] for t in y_true])


        # Compute F1 score for each threshold value
        thresholds = sorted(log_entropy)
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

        fig1.savefig(f"report/figures/latent_space_{seed}.pdf", dpi=600)
        from sklearn.preprocessing import MinMaxScaler

        # create scaler object
        scaler = MinMaxScaler(feature_range=(0, 1))

        # assume data is stored in a numpy array called 'data'
        # create a mask that identifies elements above and below the threshold
        mask = log_entropy >= optimal_threshold

        # create a copy of the data array to work with
        scaled_data = np.copy(log_entropy)

        # apply different scaling factors to the elements above and below the threshold
        scaled_data[~mask] = scaler.fit_transform(scaled_data[~mask].reshape(-1, 1)).ravel() * 0.5
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
        fig2.savefig(f"report/figures/predicted_entropy_{seed}.pdf", dpi=600)


        im = ax2.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],marker = class_markers, c=scaled_data, cmap='viridis', s=4)

        # Add legend
        ax2.legend(loc="upper left")
        ax2.set_title("Predicted uncertainty")
        ax2.set_xlabel("t-SNE Component 1")
        ax2.set_ylabel("t-SNE Component 2")

        # Add color bar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("Predicted uncertainty")

        # Save the figures
        fig2.savefig(f"report/figures/predicted_entropy_with_markers{seed}.pdf", dpi=600)

        # export the embeddings along with the labels and log entropy to a csv
        df = pd.DataFrame({'x': embeddings_tsne[:, 0], 'y': embeddings_tsne[:, 1], 'label': y_true, 'log_entropy': scaled_data})
        df.to_csv(f'data/tsne_{self.hparams.seed}.csv', index=True)
        



    def compute_auc_roc(self, ood_dataset):
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
        y_true, y_score_epistemic, y_score_aleatoric = self.compute_epistemic_aleatoric(ood_dataset)

        # Compute AUC-PR and AUC-ROC for aleatoric and epistemic confidence
        aleatoric_confidence_auc_pr = AUCPR(compute_on_step=False)( torch.tensor(y_score_aleatoric),torch.tensor(y_true))
        aleatoric_confidence_auc_roc = AUROC(compute_on_step=False)( torch.tensor(y_score_aleatoric),torch.tensor(y_true))
        epistemic_confidence_auc_pr = AUCPR(compute_on_step=False)( torch.tensor(y_score_epistemic),torch.tensor(y_true))
        epistemic_confidence_auc_roc = AUROC(compute_on_step=False)( torch.tensor(y_score_epistemic),torch.tensor(y_true))

        return (aleatoric_confidence_auc_pr, aleatoric_confidence_auc_roc, epistemic_confidence_auc_pr, epistemic_confidence_auc_roc)

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "slides.yaml")
    _ = hydra.utils.instantiate(cfg)

# Segformer model for PyTorch using Huggingface Transformer library


