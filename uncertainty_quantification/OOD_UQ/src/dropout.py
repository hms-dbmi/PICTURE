from src.models.dropout_module import DropoutModule
import torch
optimizer = torch.optim.Adam
model = DropoutModule(number_of_classes=5, optimizer = optimizer).to('cuda:0')

from src.datamodules.vienna_datamodule import ViennaDataModule

datamodule = ViennaDataModule(batch_size=32, num_workers=6)
datamodule.setup()

from pytorch_lightning import Trainer

trainer = Trainer(gpus=1, precision=16, max_epochs=2)
#trainer.fit(model, datamodule=datamodule)
trainer.validate(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
