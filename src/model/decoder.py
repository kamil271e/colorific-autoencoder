from torch import nn, cat


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
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        return x


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
