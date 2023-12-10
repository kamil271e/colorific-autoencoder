import sys
import torch
import pytorch_lightning as L
from torch import nn

sys.path.append("..")

from src.model.unet import UNet
from src.processor.processor import DataModule


class AutoEncoder(L.LightningModule):
    def __init__(self, in_channels=1, out_channels=3, unit=16):
        super().__init__()
        self.model = UNet(in_channels, out_channels, unit)
        self.criterion = nn.MSELoss()
        self.lr = 1e-3
        self.data_module = DataModule()

    def forward(self, X):
        return self.model.forward(X)

    #####################################################

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        loss = self.criterion(y, y_hat)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        loss = self.criterion(y, y_hat)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        loss = self.criterion(y, y_hat)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    #####################################################

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    #####################################################

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    #####################################################
