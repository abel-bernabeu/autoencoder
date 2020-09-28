import torch
import torch.nn as nn
from pyramidal.blocks.encoders.input import InputBlock
from pyramidal.blocks.encoders.conv import ConvBlock
from pyramidal.blocks.decoders.transpose_conv import TransposeConvBlock
from pyramidal.blocks.decoders.output import OutputBlock


class Pyramidal51CatAutoencoder(nn.Module):
    """Basic pyramidal autoencoder: 5 levels, 1 convs/level, connections by torch.cat instead of +"""

    def __init__(self, num_channels_provider, activation_factory):
        super().__init__()

        channels_0 = num_channels_provider.get_num_channels(0)
        channels_1 = num_channels_provider.get_num_channels(1)
        channels_2 = num_channels_provider.get_num_channels(2)
        channels_3 = num_channels_provider.get_num_channels(3)
        channels_4 = num_channels_provider.get_num_channels(4)
        channels_5 = num_channels_provider.get_num_channels(5)

        self.enc0 = InputBlock(3, channels_0, 3, 1, 1, activation_factory)
        self.enc1 = ConvBlock(channels_0, channels_1, 3, 2, 1, activation_factory)
        self.enc2 = ConvBlock(channels_1, channels_2, 3, 2, 1, activation_factory)
        self.enc3 = ConvBlock(channels_2, channels_3, 3, 2, 1, activation_factory)
        self.enc4 = ConvBlock(channels_3, channels_4, 3, 2, 1, activation_factory)
        self.enc5 = ConvBlock(channels_4, channels_5, 3, 2, 1, activation_factory)

        self.dec5 = TransposeConvBlock(channels_5, channels_4, 3, 2, 1, 1, activation_factory)
        self.dec4 = TransposeConvBlock(channels_4 * 2, channels_3, 3, 2, 1, 1, activation_factory)
        self.dec3 = TransposeConvBlock(channels_3 * 2, channels_2, 3, 2, 1, 1, activation_factory)
        self.dec2 = TransposeConvBlock(channels_2 * 2, channels_1, 3, 2, 1, 1, activation_factory)
        self.dec1 = TransposeConvBlock(channels_1 * 2, channels_0, 3, 2, 1, 1, activation_factory)
        self.dec0 = OutputBlock(channels_0 * 2, 3, 3, 1, 1)

    def forward(self, x):
        x0 = self.enc0(x)

        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        y5 = x5

        y4 = torch.cat((x4, self.dec5(y5)), dim=1)
        y3 = torch.cat((x3, self.dec4(y4)), dim=1)
        y2 = torch.cat((x2, self.dec3(y3)), dim=1)
        y1 = torch.cat((x1, self.dec2(y2)), dim=1)
        y0 = torch.cat((x0, self.dec1(y1)), dim=1)

        y = self.dec0(y0)

        return y
