import torch.nn as nn
from pyramidal.blocks.encoders.input import InputBlock
from pyramidal.blocks.encoders.conv import ConvBlock
from pyramidal.blocks.decoders.transpose_conv import TransposeConvBlock
from pyramidal.blocks.decoders.output import OutputBlock


class Basic22Autoencoder(nn.Module):
    """Basic non pyramidal autoencoder: 2 levels, 2 convs/level"""

    def __init__(self, num_channels_provider, activation_factory):
        super().__init__()

        channels_0 = num_channels_provider.get_num_channels(0)
        channels_1 = num_channels_provider.get_num_channels(1)
        channels_2 = num_channels_provider.get_num_channels(2)

        self.encoder = nn.Sequential(
            InputBlock(3, channels_0, 3, 1, 1, activation_factory),
            ConvBlock(channels_0, channels_0, 3, 1, 1, activation_factory),
            ConvBlock(channels_0, channels_1, 3, 2, 1, activation_factory),
            ConvBlock(channels_1, channels_1, 3, 1, 1, activation_factory),
            ConvBlock(channels_1, channels_2, 3, 2, 1, activation_factory),
            ConvBlock(channels_2, channels_2, 3, 1, 1, activation_factory),
        )

        self.decoder = nn.Sequential(
            TransposeConvBlock(channels_2, channels_2, 3, 1, 1, 0, activation_factory),
            TransposeConvBlock(channels_2, channels_1, 3, 2, 1, 1, activation_factory),
            TransposeConvBlock(channels_1, channels_1, 3, 1, 1, 0, activation_factory),
            TransposeConvBlock(channels_1, channels_0, 3, 2, 1, 1, activation_factory),
            TransposeConvBlock(channels_0, channels_0, 3, 1, 1, 0, activation_factory),
            OutputBlock(channels_0, 3, 3, 1, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
