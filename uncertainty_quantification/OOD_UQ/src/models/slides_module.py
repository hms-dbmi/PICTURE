from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from torch import nn
from torch.autograd import forward_ad
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchvision import transforms
from torchvision.utils import make_grid
# from transformers import ViTForImageClassification, SwinModel


class SlidesModule(LightningModule):
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
        optimizer: torch.optim.Optimizer,
        net: torch.nn.Module = None
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

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
        x, y = batch
        logits = self.forward(x.float())
        loss = self.criterion(logits, y.long())
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx == 0:
            confusion_matrix = wandb.plot.confusion_matrix(preds=preds.flatten().cpu().numpy(), y_true=targets.flatten().cpu().numpy(), class_names=list(self.trainer.datamodule.data_train.reduced_labels.values()))
            self.logger.experiment.log({"confusion_matrix": confusion_matrix})
        #     random_image_id = np.random.randint(0,len(batch[0]))
        #     image = batch[0][random_image_id]
        #     attention_map = self.net.visualize_attention(batch[0], grid=True)[random_image_id]
            
        #     self.logger.experiment.log({"Attention map": plt.imshow(attention_map.T)})
        #     self.logger.experiment.log({"Input": plt.imshow(image.T)})


        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

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


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "slides.yaml")
    _ = hydra.utils.instantiate(cfg)

# Segformer model for PyTorch using Huggingface Transformer library

class SegformerForSegmentation(nn.Module):
    """ Segformer model for image segmentation. The model is based on the Segformer architecture from the paper [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)."""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = SegformerModel.from_pretrained('microsoft/segformer-publaynet')
        self._init_weights(self.classifier)

    def forward(self,x):
        x = self.model(x)
        return x

    def predict(self, x):
        """Predicts the segmentation mask for a given image.
        
        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Segmentation mask of shape (B, H, W)
        """
        x = self.forward(x)
        x = torch.argmax(x, dim=1)
        return x


class ViT8(nn.Module):
    def __init__(self, type: str = "vienna", num_class: int = None, locked_encoder: bool = False):
        """ViT8 model for brain or slide images.
        
        Args:
            type (str, optional): Type of images. Defaults to "brain".
        """

        super().__init__()
        
        paths = {"brain": "/home/raa006/dino/checkpoints/brain_slide_checkpoint/run-20221010_023739-27ius0dz/files/checkpoint.pth", 
                 "cellphone": "/home/raa006/dino/checkpoints/cellphone_slide_checkpoint/checkpoint-16.pth", 
                 "brain-blurred": "/home/raa006/dino/checkpoints/brain_blured_slide_checkpoint/checkpoint.pth",
                 "whole-slide": "/home/raa006/dino/checkpoints/whole_slide/files/checkpoint.pth", 
                 "brain-test": '/home/raa006/dino/checkpoints/brain_slide_TEST_checkpoint/checkpoint.pth',
                 "vienna": "/home/raa006/dino/checkpoints/brain_slide_checkpoint/run-20221010_023739-27ius0dz/files/checkpoint.pth"}
        self.checkpoint = torch.load(paths[type])['teacher']
        self.checkpoint = {k.replace("module.", ""): v for k, v in self.checkpoint.items()}
        self.checkpoint = {k.replace("backbone.", ""): v for k, v in self.checkpoint.items()}
        
        output_size = num_class if num_class else {"brain": 10, "cellphone": 3, "brain-blurred": 10, "whole-slide": 10, "brain-test": 10, "vienna": 2}[type] 
        
        self.net = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        #self.net.load_state_dict(self.checkpoint, strict=False)
        self.mlp_head = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(384),
            nn.Linear(384, output_size)
        )
        
    def visualize_attention(self, x, grid=False, quantile = 0):
        self.net.eval()
        with torch.no_grad():
            image = transforms.Resize((480,480))(x)

            w_featmap = image.shape[-2] // 8
            h_featmap = image.shape[-1] // 8

            attentions = self.net.get_last_selfattention(image.float())

            nh = attentions.shape[1] # number of head
            nb = image.shape[0] # batch size
            
            # we keep only the output patch attention
            attentions = attentions[:, :, 0, 1:].reshape(nb, nh, -1)

            attentions = attentions.reshape(nb, nh, w_featmap, h_featmap)
            #attentions = nn.functional.interpolate(attentions, scale_factor=8, mode="nearest")
            
            # threshold each image in attentions by the quantile
            if quantile:
                for i in range(attentions.shape[0]):
                    for j in range(attentions.shape[1]):
                        attentions[i,j] = torch.where(attentions[i,j] > torch.quantile(attentions[i,j], quantile),  torch.quantile(attentions[i,j], quantile), attentions[i,j])

            
        if grid:
            all_grids = []
            for b in range(x.shape[0]):
                grid = make_grid([torch.unsqueeze(head,0) for head in attentions[b]],nrow=2, normalize=True, scale_each=True, pad_value=1)
                all_grids.append(grid)
            attentions= torch.stack(all_grids)
        
        return attentions.cpu()
                
    def forward(self, x):
        x = self.net(x)
        x = self.mlp_head(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = torch.argmax(x, dim=1)
        return x

