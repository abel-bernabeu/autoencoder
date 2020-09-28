import torch.nn as nn
from pyramidal.blocks.densenet.down_factory import AbstractDownFactory


class DownAvgPool(nn.Module):
    """Downsampler module with average pool"""

    def __init__(self, channels):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.AvgPool2d(2, 2, 0),
        )

        self.channels = channels

    def forward(self, x):
        return self.network(x)

    def get_in_channels(self):
        return self.channels

    def get_out_channels(self):
        return self.channels


class DownAvgPoolFactory(AbstractDownFactory):
    """Downsampler factory: returns DownAvgPool modules"""

    def get_down(self, channels):
        return DownAvgPool(channels)

