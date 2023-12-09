from torch import nn, cat
from torch.nn.functional import relu
import pytorch_lightning as L


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x_pooled = self.pool(x)
        return x, x_pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=upsample_factor
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = cat([x, skip_connection], dim=1)
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, unit):
        super(Encoder, self).__init__()
        self.enc1 = EncoderBlock(in_channels, unit)
        self.enc2 = EncoderBlock(unit, 2 * unit)
        self.enc3 = EncoderBlock(2 * unit, 4 * unit)
        self.enc4 = EncoderBlock(4 * unit, 8 * unit)
        self.enc5 = EncoderBlock(8 * unit, 16 * unit)

    def forward(self, x):
        xe1, xp1 = self.enc1(x)
        xe2, xp2 = self.enc2(xp1)
        xe3, xp3 = self.enc3(xp2)
        xe4, xp4 = self.enc4(xp3)
        xe5, _ = self.enc5(xp4)
        return [xe1, xe2, xe3, xe4, xe5]


class Decoder(nn.Module):
    def __init__(self, out_channels, unit):
        super(Decoder, self).__init__()
        self.dec0 = DecoderBlock(16 * unit, 8 * unit, 2)
        self.dec1 = DecoderBlock(8 * unit, 4 * unit, 2)
        self.dec2 = DecoderBlock(4 * unit, 2 * unit, 2)
        self.dec3 = DecoderBlock(2 * unit, unit, 2)
        self.outconv = nn.Conv2d(unit, out_channels, kernel_size=1)

    def forward(self, encoded_features):
        xd0 = self.dec0(encoded_features[4], encoded_features[3])
        xd1 = self.dec1(xd0, encoded_features[2])
        xd2 = self.dec2(xd1, encoded_features[1])
        xd3 = self.dec3(xd2, encoded_features[0])
        out = self.outconv(xd3)
        return out


class UNet(L.LightningModule):
    def __init__(self, in_channels, out_channels, unit=16):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, unit)
        self.decoder = Decoder(out_channels, unit)

    def forward(self, X):
        encoded_features = self.encoder(X)
        out = self.decoder(encoded_features)
        return out
