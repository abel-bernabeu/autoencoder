import torch
import torch.nn as nn
from pyramidal.blocks.densenet.layer import Layer


class DenseBlock7(nn.Module):
    """DenseNet Block of 7 layers"""

    def __init__(self, in_channels, k):
        super().__init__()

        self.layer1 = Layer(in_channels, k)
        self.layer2 = Layer(in_channels + k, k)
        self.layer3 = Layer(in_channels + k * 2, k)
        self.layer4 = Layer(in_channels + k * 3, k)
        self.layer5 = Layer(in_channels + k * 4, k)
        self.layer6 = Layer(in_channels + k * 5, k)
        self.layer7 = Layer(in_channels + k * 6, k)

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
        return torch.cat((x1, x2, x3, x4, x5, x6, x7), dim=1)

    def get_out_channels(self):
        return 7 * self.k