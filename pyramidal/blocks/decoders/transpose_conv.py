import torch.nn as nn


class TransposeConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, output_padding, activation_factory):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding=output_padding),
            nn.BatchNorm2d(output_channels),
            activation_factory.get_activation(),
        )

    def forward(self, x):
        return self.network(x)