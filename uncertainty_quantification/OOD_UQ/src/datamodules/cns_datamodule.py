from cgi import test
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from src.datasets.cns_dataset import CNSDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CNSDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        slides_file: str = "/n/data2/hms/dbmi/kyu/lab/jz290/BRAIN_20X/",
        train_val_test_split: Tuple[int, int, int] = (0.5, 0.3, 0.2),
        batch_size: int = 8,
        num_workers: int = 6,
        pin_memory: bool = False,
        dino: bool = False,
        transforms = None,
        target_transform = None,
        p_uncertainty: float = 0.0,
        normalize_stain: bool = False,
        label_to_use = None,
        label_map = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        
        if not transforms:
            transforms = [A.Resize(224,224)]
        self.transforms = A.Compose(transforms)


        if not target_transform:
            target_transform = []
        self.target_transform = A.Compose(target_transform+[ToTensorV2(), A.Resize(224,224)])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        

    @property
    def num_classes(self):
        return len(set(self.data_train.label_map.values()))

    @property
    def output_type(self):
        return "categorical"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([3, 224, 224])

    @property
    def gradient_accumulation_steps(self) -> int:
        """
        Returns the number of batches from which to accumulate the gradients for training.
        """
        return 1

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        return None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = CNSDataset(slides_file=self.hparams.slides_file, target_transform = self.target_transform, transform= self.transforms, dino = self.hparams.dino, normalize_stain=self.hparams.normalize_stain, label_map = self.hparams.label_map)


            total_samples = len(dataset)
            train_samples = int(self.hparams.train_val_test_split[0]*total_samples)
            val_samples = int(self.hparams.train_val_test_split[1]*total_samples)
            test_samples = total_samples-train_samples-val_samples
            
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=(train_samples,val_samples,test_samples),
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "slides.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
