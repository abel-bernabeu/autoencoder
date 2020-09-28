import torch
import torch.nn as nn
from pyramidal.blocks.encoders.input import InputBlock
from pyramidal.blocks.decoders.output import OutputBlock
from pyramidal.blocks.activations.relu import ReLUFactory
from pyramidal.blocks.densenet.denseblock4 import DenseBlock4
from pyramidal.blocks.densenet.denseblock5 import DenseBlock5
from pyramidal.blocks.densenet.denseblock7 import DenseBlock7
from pyramidal.blocks.densenet.down_maxpool import DownMaxPool
from pyramidal.blocks.densenet.up_conv_transpose import UpConvTranspose


class Pyramidal457DenseNetAutoencoder(nn.Module):
    """Basic DenseNet-like pyramidal autoencoder: 2 levels, 4 layers at level 1, 5 layers at level 2, 7 layers at bottleneck"""

    def __init__(self, num_channels, k):
        super().__init__()

        activation_factory = ReLUFactory()

        self.input = InputBlock(3, num_channels, 3, 1, 1, activation_factory)
        self.enc1 = DenseBlock4(num_channels, k)
        self.down1 = DownMaxPool(num_channels + self.enc1.get_out_channels())
        self.enc2 = DenseBlock5(num_channels + self.enc1.get_out_channels(), k)
        self.down2 = DownMaxPool(num_channels + self.enc1.get_out_channels() + self.enc2.get_out_channels())

        self.dense = DenseBlock7(num_channels + self.enc1.get_out_channels() + self.enc2.get_out_channels(), k)

        self.up2 = UpConvTranspose(self.dense.get_out_channels())
        self.dec2 = DenseBlock5(num_channels + self.enc1.get_out_channels() + self.enc2.get_out_channels() + self.dense.get_out_channels(), k)
        self.up1 = UpConvTranspose(self.dec2.get_out_channels())
        self.dec1 = DenseBlock4(num_channels + self.enc1.get_out_channels() + self.dec2.get_out_channels(), k)
        self.output = OutputBlock(self.dec1.get_out_channels(), 3, 3, 1, 1)

    def forward(self, x):
        x1 = self.input(x)                  # num_channels
        x2 = self.enc1(x1)                  # self.enc1.get_out_channels()
        x3 = torch.cat((x1, x2), dim=1)     # num_channels + self.enc1.get_out_channels()
        x4 = self.down1(x3)                 # num_channels + self.enc1.get_out_channels()
        x5 = self.enc2(x4)                  # self.enc2.get_out_channels()
        x6 = torch.cat((x4, x5), dim=1)     # num_channels + self.enc1.get_out_channels() + self.enc2.get_out_channels()
        x7 = self.down2(x6)                 # num_channels + self.enc1.get_out_channels() + self.enc2.get_out_channels()

        y7 = self.dense(x7)                 # self.dense.get_out_channels()
        y6 = self.up2(y7)                   # self.dense.get_out_channels()
        y5 = torch.cat((x6, y6), dim=1)     # num_channels + self.enc1.get_out_channels() + self.enc2.get_out_channels() + self.dense.get_out_channels()
        y4 = self.dec2(y5)                  # self.dec2.get_out_channels()
        y3 = self.up1(y4)                   # self.dec2.get_out_channels()
        y2 = torch.cat((x3, y3), dim=1)     # num_channels + self.enc1.get_out_channels() + self.dec2.get_out_channels()
        y1 = self.dec1(y2)                  # self.dec1.get_out_channels()
        y = self.output(y1)                 # 3

        return y