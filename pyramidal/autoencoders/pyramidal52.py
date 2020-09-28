import torch.nn as nn
from pyramidal.blocks.encoders.input import InputBlock
from pyramidal.blocks.encoders.conv import ConvBlock
from pyramidal.blocks.decoders.transpose_conv import TransposeConvBlock
from pyramidal.blocks.decoders.output import OutputBlock


class Pyramidal52Autoencoder(nn.Module):
    """Basic pyramidal autoencoder: 5 levels, 2 convs/level"""

    def __init__(self, num_channels_provider, activation_factory):
        super().__init__()

        channels_0 = num_channels_provider.get_num_channels(0)
        channels_1 = num_channels_provider.get_num_channels(1)
        channels_2 = num_channels_provider.get_num_channels(2)
        channels_3 = num_channels_provider.get_num_channels(3)
        channels_4 = num_channels_provider.get_num_channels(4)
        channels_5 = num_channels_provider.get_num_channels(5)

        self.enc0_ = InputBlock(3, channels_0, 3, 1, 1, activation_factory)
        self.enc0b = ConvBlock(channels_0, channels_0, 3, 1, 1, activation_factory)
        self.enc1_ = ConvBlock(channels_0, channels_1, 3, 2, 1, activation_factory)
        self.enc1b = ConvBlock(channels_1, channels_1, 3, 1, 1, activation_factory)
        self.enc2_ = ConvBlock(channels_1, channels_2, 3, 2, 1, activation_factory)
        self.enc2b = ConvBlock(channels_2, channels_2, 3, 1, 1, activation_factory)
        self.enc3_ = ConvBlock(channels_2, channels_3, 3, 2, 1, activation_factory)
        self.enc3b = ConvBlock(channels_3, channels_3, 3, 1, 1, activation_factory)
        self.enc4_ = ConvBlock(channels_3, channels_4, 3, 2, 1, activation_factory)
        self.enc4b = ConvBlock(channels_4, channels_4, 3, 1, 1, activation_factory)
        self.enc5_ = ConvBlock(channels_4, channels_5, 3, 2, 1, activation_factory)
        self.enc5b = ConvBlock(channels_5, channels_5, 3, 1, 1, activation_factory)

        self.dec5b = TransposeConvBlock(channels_5, channels_5, 3, 1, 1, 0, activation_factory)
        self.dec5_ = TransposeConvBlock(channels_5, channels_4, 3, 2, 1, 1, activation_factory)
        self.dec4b = TransposeConvBlock(channels_4, channels_4, 3, 1, 1, 0, activation_factory)
        self.dec4_ = TransposeConvBlock(channels_4, channels_3, 3, 2, 1, 1, activation_factory)
        self.dec3b = TransposeConvBlock(channels_3, channels_3, 3, 1, 1, 0, activation_factory)
        self.dec3_ = TransposeConvBlock(channels_3, channels_2, 3, 2, 1, 1, activation_factory)
        self.dec2b = TransposeConvBlock(channels_2, channels_2, 3, 1, 1, 0, activation_factory)
        self.dec2_ = TransposeConvBlock(channels_2, channels_1, 3, 2, 1, 1, activation_factory)
        self.dec1b = TransposeConvBlock(channels_1, channels_1, 3, 1, 1, 0, activation_factory)
        self.dec1_ = TransposeConvBlock(channels_1, channels_0, 3, 2, 1, 1, activation_factory)
        self.dec0b = TransposeConvBlock(channels_0, channels_0, 3, 1, 1, 0, activation_factory)
        self.dec0_ = OutputBlock(channels_0, 3, 3, 1, 1)

    def forward(self, x):
        x0 = self.enc0b(self.enc0_(x))

        x1 = self.enc1b(self.enc1_(x0))
        x2 = self.enc2b(self.enc2_(x1))
        x3 = self.enc3b(self.enc3_(x2))
        x4 = self.enc4b(self.enc4_(x3))
        x5 = self.enc5b(self.enc5_(x4))

        y5 = x5

        y4 = x4 + self.dec5_(self.dec5b(y5))
        y3 = x3 + self.dec4_(self.dec4b(y4))
        y2 = x2 + self.dec3_(self.dec3b(y3))
        y1 = x1 + self.dec2_(self.dec2b(y2))
        y0 = x0 + self.dec1_(self.dec1b(y1))

        y = self.dec0_(self.dec0b(y0))

        return y
