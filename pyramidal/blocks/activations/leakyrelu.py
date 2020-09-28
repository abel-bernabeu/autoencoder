from pyramidal.blocks.activations.base import AbstractActivationFactory
import torch.nn as nn


class LeakyReLUFactory(AbstractActivationFactory):
    """Activation function factory: returns LeakyReLU"""

    def __init__(self, negative_slope=0.01, inplace=False):
        self.negative_slope = negative_slope
        self.inplace = inplace

    def get_activation(self):
        return nn.LeakyReLU(negative_slope=self.negative_slope, inplace=self.inplace)