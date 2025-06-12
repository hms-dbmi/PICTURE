from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.autograd import forward_ad
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchvision import transforms
from torchvision.utils import make_grid
from transformers import ViTForImageClassification, SwinModel


class SegmentationModule(LightningModule):
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
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

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
            random_image_id = np.random.randint(0,len(batch[0]))
            image = batch[0][random_image_id]
            attention_map = self.net.visualize_attention(batch[0], grid=True)[random_image_id]
            
            self.logger.experiment.log({"Attention map": plt.imshow(attention_map.T)})
            self.logger.experiment.log({"Input": plt.imshow(image.T)})


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
    def __init__(self, type: str = "brain"):
        """ViT8 model for brain or slide images.
        
        Args:
            type (str, optional): Type of images. Defaults to "brain".
        """

        super().__init__()
        
        paths = {"brain": "/home/raa006/dino/checkpoints/brain_slide_checkpoint/run-20221010_023739-27ius0dz/files/checkpoint.pth", 
                 "cellphone": "/home/raa006/dino/checkpoints/cellphone_slide_checkpoint/checkpoint-16.pth", 
                 "brain-blurred": "/home/raa006/dino/checkpoints/brain_blured_slide_checkpoint/checkpoint.pth",
                 "whole-slide": "/home/raa006/dino/checkpoints/whole_slide/files/checkpoint.pth"}
        self.checkpoint = torch.load(paths[type])['teacher']
        self.checkpoint = {k.replace("module.", ""): v for k, v in self.checkpoint.items()}
        self.checkpoint = {k.replace("backbone.", ""): v for k, v in self.checkpoint.items()}
        
        output_size = {"brain": 10, "cellphone": 3, "brain-blurred": 10, "whole-slide": 10}[type]
        
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


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size = 224, patch_size = 8, num_classes = 5, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)