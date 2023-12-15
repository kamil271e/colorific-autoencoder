import sys
import pytorch_lightning as L

sys.path.append("..")

from src.models.unet.encoder import Encoder
from src.models.unet.decoder import Decoder


class UNet(L.LightningModule):
    def __init__(self, in_channels, out_channels, unit):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, unit)
        self.decoder = Decoder(out_channels, unit)

    def forward(self, X):
        encoded_features = self.encoder(X)
        out = self.decoder(encoded_features).sigmoid()
        return out
