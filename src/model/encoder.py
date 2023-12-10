from torch import nn


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
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x_pooled = self.pool(x)
        return x, x_pooled


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
