import torch.nn as nn
from pyramidal.blocks.densenet.up_factory import AbstractUpFactory


class UpConvTransposeExt(nn.Module):
    """Upsampler module with transposed convolution plus batch normalization, relu and dropout"""

    def __init__(self, channels, n):
        super().__init__()

        self.network = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels // n, 3, 2, 1, output_padding=1),
            nn.Dropout(p=0.2),
        )

        self.channels = channels
        self.n = n

    def forward(self, x):
        return self.network(x)

    def get_in_channels(self):
        return self.channels

    def get_out_channels(self):
        return self.channels // self.n


class UpConvTransposeExtFactory(AbstractUpFactory):
    """Upsampler factory: returns UpConvTransposeExt modules"""

    def __init__(self, n):
        self.n = n

    def get_up(self, channels):
        return UpConvTransposeExt(channels, self.n)