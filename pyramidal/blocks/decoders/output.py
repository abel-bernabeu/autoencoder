import torch.nn as nn


class OutputBlock(nn.Module):
    """Transposed Convolutional block: convolution plus Tanh. Intended to be the last block of the autoencoder.
    The output will range from -1 to 1. Input should be the same range."""

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)