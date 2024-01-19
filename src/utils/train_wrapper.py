import sys
import torch
import pytorch_lightning as L
import matplotlib.pyplot as plt
from torch import nn

sys.path.append("..")

from src.models.unet.unet import UNet
from src.processor.processor import DataModule
from src.utils.config import Config

config = Config()


class AutoEncoder(L.LightningModule):
    def __init__(
        self,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        unit=config.OUT_CHANNELS,
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        dropout_rate=config.DROPOUT_RATE,
    ):
        super().__init__()
        self.model = UNet(in_channels, out_channels, unit, dropout_rate)
        self.criterion = nn.L1Loss()
        self.lr = lr
        self.weight_decay = weight_decay
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
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        loss = self.criterion(y, y_hat)
        self.total_val_loss_epoch += loss
        self.val_samples_epoch += X.size(0)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        loss = self.criterion(y, y_hat)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        print("WEIGHT_DECAY:", self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.SCHEDULER_STEP_SIZE,
            gamma=config.SCHEDULER_GAMMA,
        )
        return [optimizer], [scheduler]

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
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)

    def plot_loss(self):
        x = torch.arange(1, len(self.train_loss_list) + 1)
        plt.figure(figsize=(12, 8))
        plt.plot(x, self.train_loss_list, label="train loss")
        plt.plot(x, self.val_loss_list, label="val loss")
        plt.title("Loss plot")
        plt.legend()
        plt.show()

    def visualize_predict(self, num_samples, batches=1):

        for i, batch in enumerate(self.test_dataloader()):
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
            if i - 1 >= batches:
                break

    #####################################################

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    #####################################################

    def load_model(self, path=config.CKPT_PATH):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt["state_dict"])
