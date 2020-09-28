import torch.nn as nn


class InputBlock(nn.Module):
    """Convolutional block: convolution plus activation. Intended to be the first block of the autoencoder."""

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, activation_factory):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            activation_factory.get_activation(),
        )

    def forward(self, x):
        return self.network(x)