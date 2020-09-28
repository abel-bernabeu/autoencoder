import torch.nn as nn


class Layer(nn.Module):
    """Basic building unit of the DenseNet autoencoder arquitecture"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.network = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        return self.network(x)