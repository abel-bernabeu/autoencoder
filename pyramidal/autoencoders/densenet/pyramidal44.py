import torch
import torch.nn as nn
from pyramidal.blocks.encoders.input import InputBlock
from pyramidal.blocks.decoders.output import OutputBlock
from pyramidal.blocks.activations.relu import ReLUFactory
from pyramidal.blocks.densenet.denseblock4 import DenseBlock4
from pyramidal.blocks.densenet.down_maxpool import DownMaxPool
from pyramidal.blocks.densenet.up_conv_transpose import UpConvTranspose


class Pyramidal44DenseNetAutoencoder(nn.Module):
    """Basic DenseNet-like pyramidal autoencoder: 1 levels, 4 layers at level 1, 4 layers at bottleneck"""

    def __init__(self, num_channels, k):
        super().__init__()

        activation_factory = ReLUFactory()

        self.input = InputBlock(3, num_channels, 3, 1, 1, activation_factory)
        self.enc1 = DenseBlock4(num_channels, k)
        self.down1 = DownMaxPool(num_channels + self.enc1.get_out_channels())

        self.dense = DenseBlock4(num_channels + self.enc1.get_out_channels(), k)

        self.up1 = UpConvTranspose(self.dense.get_out_channels())
        self.dec1 = DenseBlock4(num_channels + self.enc1.get_out_channels() + self.dense.get_out_channels(), k)
        self.output = OutputBlock(self.dec1.get_out_channels(), 3, 3, 1, 1)

    def forward(self, x):
        x1 = self.input(x)                  # num_channels
        x2 = self.enc1(x1)                  # self.enc1.get_out_channels()
        x3 = torch.cat((x1, x2), dim=1)     # num_channels + self.enc1.get_out_channels()
        x4 = self.down1(x3)                 # num_channels + self.enc1.get_out_channels()

        y4 = self.dense(x4)                 # self.dense.get_out_channels()
        y3 = self.up1(y4)                   # self.dense.get_out_channels()
        y2 = torch.cat((x3, y3), dim=1)     # num_channels + self.enc1.get_out_channels() + self.dense.get_out_channels()
        y1 = self.dec1(y2)                  # self.dec1.get_out_channels()
        y = self.output(y1)                 # 3

        return y