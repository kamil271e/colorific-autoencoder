import sys
import pytorch_lightning as L

sys.path.append("..")

from src.model.encoder import Encoder
from src.model.decoder import Decoder


class UNet(L.LightningModule):
    def __init__(self, in_channels, out_channels, unit=16):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, unit)
        self.decoder = Decoder(out_channels, unit)

    def forward(self, X):
        encoded_features = self.encoder(X)
        out = self.decoder(encoded_features).sigmoid()
        return out

    #####################################################
    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    #####################################################

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    #####################################################

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    #####################################################
