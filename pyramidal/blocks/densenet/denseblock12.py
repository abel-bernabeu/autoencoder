import torch
import torch.nn as nn
from pyramidal.blocks.densenet.layer import Layer


class DenseBlock12(nn.Module):
    """DenseNet Block of 12 layers"""

    def __init__(self, in_channels, k):
        super().__init__()

        self.layer1 = Layer(in_channels, k)
        self.layer2 = Layer(in_channels + k, k)
        self.layer3 = Layer(in_channels + k * 2, k)
        self.layer4 = Layer(in_channels + k * 3, k)
        self.layer5 = Layer(in_channels + k * 4, k)
        self.layer6 = Layer(in_channels + k * 5, k)
        self.layer7 = Layer(in_channels + k * 6, k)
        self.layer8 = Layer(in_channels + k * 7, k)
        self.layer9 = Layer(in_channels + k * 8, k)
        self.layer10 = Layer(in_channels + k * 9, k)
        self.layer11 = Layer(in_channels + k * 10, k)
        self.layer12 = Layer(in_channels + k * 11, k)

        self.k = k

    def forward(self, x):

        x1 = self.layer1(x)
        c1 = torch.cat((x, x1), dim=1)
        x2 = self.layer2(c1)
        c2 = torch.cat((c1, x2), dim=1)
        x3 = self.layer3(c2)
        c3 = torch.cat((c2, x3), dim=1)
        x4 = self.layer4(c3)
        c4 = torch.cat((c3, x4), dim=1)
        x5 = self.layer5(c4)
        c5 = torch.cat((c4, x5), dim=1)
        x6 = self.layer6(c5)
        c6 = torch.cat((c5, x6), dim=1)
        x7 = self.layer7(c6)
        c7 = torch.cat((c6, x7), dim=1)
        x8 = self.layer8(c7)
        c8 = torch.cat((c7, x8), dim=1)
        x9 = self.layer9(c8)
        c9 = torch.cat((c8, x9), dim=1)
        x10 = self.layer10(c9)
        c10 = torch.cat((c9, x10), dim=1)
        x11 = self.layer11(c10)
        c11 = torch.cat((c10, x11), dim=1)
        x12 = self.layer12(c11)
        return torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)

    def get_out_channels(self):
        return 12 * self.k