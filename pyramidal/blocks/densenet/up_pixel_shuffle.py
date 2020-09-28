import torch.nn as nn
from pyramidal.blocks.densenet.up_factory import AbstractUpFactory


class UpPixelShuffle(nn.Module):
    """Upsampler module with PixelShuffle (experimental: probably needs convolution after pixel shuffle for controlling number of output channels)."""

    def __init__(self, channels):
        super().__init__()

        self.network = nn.PixelShuffle(2)

        self.channels = channels

    def forward(self, x):
        return self.network(x)

    def get_in_channels(self):
        return self.channels

    def get_out_channels(self):
        return self.channels // 4


class UpPixelShuffleFactory(AbstractUpFactory):
    """Upsampler factory: returns UpPixelShaffle modules"""

    def get_up(self, channels):
        return UpPixelShuffle(channels)