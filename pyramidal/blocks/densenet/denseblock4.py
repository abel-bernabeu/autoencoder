import torch
import torch.nn as nn
from pyramidal.blocks.densenet.layer import Layer


class DenseBlock4(nn.Module):
    """DenseNet Block of 4 layers"""

    def __init__(self, in_channels, k):
        super().__init__()

        self.layer1 = Layer(in_channels, k)
        self.layer2 = Layer(in_channels + k, k)
        self.layer3 = Layer(in_channels + k * 2, k)
        self.layer4 = Layer(in_channels + k * 3, k)

        self.k = k

    def forward(self, x):

        x1 = self.layer1(x)
        c1 = torch.cat((x, x1), dim=1)
        x2 = self.layer2(c1)
        c2 = torch.cat((c1, x2), dim=1)
        x3 = self.layer3(c2)
        c3 = torch.cat((c2, x3), dim=1)
        x4 = self.layer4(c3)
        return torch.cat((x1, x2, x3, x4), dim=1)

    def get_out_channels(self):
        return 4 * self.k