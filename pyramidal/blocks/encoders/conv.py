import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block: convolution plus batch normalization plus activation"""

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, activation_factory):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels),
            activation_factory.get_activation(),
        )

    def forward(self, x):
        return self.network(x)