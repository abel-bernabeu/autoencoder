import torch.nn as nn
from pyramidal.blocks.densenet.up_factory import AbstractUpFactory


class UpConvTranspose(nn.Module):
    """Upsampler module with transposed convolution"""

    def __init__(self, channels):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 3, 2, 1, output_padding=1),
        )

        self.channels = channels

    def forward(self, x):
        return self.network(x)

    def get_in_channels(self):
        return self.channels

    def get_out_channels(self):
        return self.channels


class UpConvTransposeFactory(AbstractUpFactory):
    """Upsampler factory: returns UpConvTranspose modules"""

    def get_up(self, channels):
        return UpConvTranspose(channels)