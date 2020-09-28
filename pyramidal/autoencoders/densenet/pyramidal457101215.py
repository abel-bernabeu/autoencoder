import torch
import torch.nn as nn
from pyramidal.blocks.encoders.input import InputBlock
from pyramidal.blocks.decoders.output import OutputBlock
from pyramidal.blocks.activations.relu import ReLUFactory
from pyramidal.blocks.densenet.denseblock4 import DenseBlock4
from pyramidal.blocks.densenet.denseblock5 import DenseBlock5
from pyramidal.blocks.densenet.denseblock7 import DenseBlock7
from pyramidal.blocks.densenet.denseblock10 import DenseBlock10
from pyramidal.blocks.densenet.denseblock12 import DenseBlock12
from pyramidal.blocks.densenet.denseblock15 import DenseBlock15
from pyramidal.blocks.densenet.down_maxpool import DownMaxPool
from pyramidal.blocks.densenet.up_conv_transpose import UpConvTranspose


class Pyramidal457101215DenseNetAutoencoder(nn.Module):
    """FINAL DenseNet-like pyramidal autoencoder: 5 levels
    4 layers at level 1
    5 layers at level 2
    7 layers at level 3
    10 layers at level 4
    12 layers at level 5
    15 layers at bottleneck"""

    def __init__(self, num_channels, k):
        super().__init__()

        activation_factory = ReLUFactory()

        self.input = InputBlock(3, num_channels, 3, 1, 1, activation_factory)
        self.enc1 = DenseBlock4(num_channels, k)
        self.down1 = DownMaxPool(num_channels + self.enc1.get_out_channels())
        self.enc2 = DenseBlock5(self.down1.get_out_channels(), k)
        self.down2 = DownMaxPool(self.down1.get_out_channels() + self.enc2.get_out_channels())
        self.enc3 = DenseBlock7(self.down2.get_out_channels(), k)
        self.down3 = DownMaxPool(self.down2.get_out_channels() + self.enc3.get_out_channels())
        self.enc4 = DenseBlock10(self.down3.get_out_channels(), k)
        self.down4 = DownMaxPool(self.down3.get_out_channels() + self.enc4.get_out_channels())
        self.enc5 = DenseBlock12(self.down4.get_out_channels(), k)
        self.down5 = DownMaxPool(self.down4.get_out_channels() + self.enc5.get_out_channels())

        self.dense = DenseBlock15(self.down5.get_out_channels(), k)

        self.up5 = UpConvTranspose(self.dense.get_out_channels())
        self.dec5 = DenseBlock12(self.down4.get_out_channels() + self.enc5.get_out_channels() + self.up5.get_out_channels(), k)
        self.up4 = UpConvTranspose(self.dec5.get_out_channels())
        self.dec4 = DenseBlock10(self.down3.get_out_channels() + self.enc4.get_out_channels() + self.up4.get_out_channels(), k)
        self.up3 = UpConvTranspose(self.dec4.get_out_channels())
        self.dec3 = DenseBlock7(self.down2.get_out_channels() + self.enc3.get_out_channels() + self.up3.get_out_channels(), k)
        self.up2 = UpConvTranspose(self.dec3.get_out_channels())
        self.dec2 = DenseBlock5(self.down1.get_out_channels() + self.enc2.get_out_channels() + self.up2.get_out_channels(), k)
        self.up1 = UpConvTranspose(self.dec2.get_out_channels())
        self.dec1 = DenseBlock5(num_channels + self.enc1.get_out_channels() + self.up1.get_out_channels(), k)
        self.output = OutputBlock(self.dec1.get_out_channels(), 3, 3, 1, 1)

    def forward(self, x):
        x1 = self.input(x)                  # num_channels
        x2 = self.enc1(x1)                  # self.enc1.get_out_channels()
        x3 = torch.cat((x1, x2), dim=1)     # num_channels + self.enc1.get_out_channels()
        x4 = self.down1(x3)                 # self.down1.get_out_channels()
        x5 = self.enc2(x4)                  # self.enc2.get_out_channels()
        x6 = torch.cat((x4, x5), dim=1)     # self.down1.get_out_channels() + self.enc2.get_out_channels()
        x7 = self.down2(x6)                 # self.down2.get_out_channels()
        x8 = self.enc3(x7)                  # self.enc3.get_out_channels()
        x9 = torch.cat((x7, x8), dim=1)     # self.down2.get_out_channels() + self.enc3.get_out_channels()
        x10 = self.down3(x9)                # self.down3.get_out_channels()
        x11 = self.enc4(x10)                # self.enc4.get_out_channels()
        x12 = torch.cat((x10, x11), dim=1)  # self.down3.get_out_channels() + self.enc4.get_out_channels()
        x13 = self.down4(x12)               # self.down4.get_out_channels()
        x14 = self.enc5(x13)                # self.enc5.get_out_channels()
        x15 = torch.cat((x13, x14), dim=1)  # self.down4.get_out_channels() + self.enc5.get_out_channels()
        x16 = self.down5(x15)               # self.down5.get_out_channels()

        y16 = self.dense(x16)               # self.dense.get_out_channels()
        y15 = self.up5(y16)                 # self.up5.get_out_channels()
        y14 = torch.cat((x15, y15), dim=1)  # self.down4.get_out_channels() + self.enc5.get_out_channels() + self.up5.get_out_channels()
        y13 = self.dec5(y14)                # self.dec5.get_out_channels()
        y12 = self.up4(y13)                 # self.up4.get_out_channels()
        y11 = torch.cat((x12, y12), dim=1)  # self.down3.get_out_channels() + self.enc4.get_out_channels() + self.up4.get_out_channels()
        y10 = self.dec4(y11)                # self.dec4.get_out_channels()
        y9 = self.up3(y10)                  # self.up3.get_out_channels()
        y8 = torch.cat((x9, y9), dim=1)     # self.down2.get_out_channels() + self.enc3.get_out_channels() + self.up3.get_out_channels()
        y7 = self.dec3(y8)                  # self.dec3.get_out_channels()
        y6 = self.up2(y7)                   # self.up2.get_out_channels()
        y5 = torch.cat((x6, y6), dim=1)     # self.down1.get_out_channels() + self.enc2.get_out_channels() + self.up2.get_out_channels()
        y4 = self.dec2(y5)                  # self.dec2.get_out_channels()
        y3 = self.up1(y4)                   # self.up1.get_out_channels()
        y2 = torch.cat((x3, y3), dim=1)     # num_channels + self.enc1.get_out_channels() + self.up1.get_out_channels()
        y1 = self.dec1(y2)                  # self.dec1.get_out_channels()
        y = self.output(y1)                 # 3

        return y