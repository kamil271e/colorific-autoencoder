import sys
import torch
import pytorch_lightning as L
import matplotlib.pyplot as plt
from torch import nn

sys.path.append("..")

from src.models.unet.unet import UNet
from src.processor.processor import DataModule


class AutoEncoder(L.LightningModule):
    def __init__(self, in_channels=1, out_channels=3, unit=16, lr=1e-3):
        super().__init__()
        self.model = UNet(in_channels, out_channels, unit)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.data_module = DataModule()

        self.total_train_loss_epoch = 0
        self.train_samples_epoch = 0
        self.train_loss_list = []
        self.total_val_loss_epoch = 0
        self.val_samples_epoch = 0
        self.val_loss_list = []

    def forward(self, X):
        return self.model.forward(X)

    #####################################################

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        loss = self.criterion(y, y_hat)
        self.total_train_loss_epoch += loss
        self.train_samples_epoch += X.size(0)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        loss = self.criterion(y, y_hat)
        self.total_val_loss_epoch += loss
        self.val_samples_epoch += X.size(0)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        loss = self.criterion(y, y_hat)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    #####################################################

    def on_train_epoch_end(self):
        avg_train_loss = self.total_train_loss_epoch / self.train_samples_epoch
        self.train_loss_list.append(avg_train_loss.cpu().detach().numpy())
        self.total_train_loss_epoch = 0
        self.train_samples_epoch = 0

    def on_validation_epoch_end(self):
        avg_val_loss = self.total_val_loss_epoch / self.val_samples_epoch
        self.val_loss_list.append(avg_val_loss.cpu().detach().numpy())
        self.total_val_loss_epoch = 0
        self.val_samples_epoch = 0

    def on_test_epoch_end(self):
        pass

    def plot_loss(self):
        x = torch.arange(1, len(self.train_loss_list) + 1)
        plt.figure(figsize=(12, 8))
        plt.plot(x, self.train_loss_list, label="train loss")
        plt.plot(x, self.val_loss_list, label="val loss")
        plt.title("Loss plot")
        plt.legend()
        plt.show()

    def visualize_predict(self, num_samples):
        batch = next(iter(self.test_dataloader()))
        X, y = batch
        y_hat = self.model.forward(X)

        fig, ax = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
        for i in range(num_samples):
            ax[i, 0].imshow(X[i][0], cmap="gray")
            ax[i, 0].set_title("Input")
            ax[i, 0].axis("off")
            ax[i, 1].imshow(y[i].permute(1, 2, 0))
            ax[i, 1].set_title("Target")
            ax[i, 1].axis("off")
            ax[i, 2].imshow(y_hat[i].permute(1, 2, 0).detach().numpy())
            ax[i, 2].set_title("Prediction")
            ax[i, 2].axis("off")
        plt.show()
        print(f"batch loss: {self.criterion(y, y_hat)}")

    #####################################################

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    #####################################################

    def save_model(self, path="checkpoint.pt"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="checkpoint.pt"):
        self.model.load_state_dict(torch.load(path))
