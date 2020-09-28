from pyramidal.blocks.activations.base import AbstractActivationFactory
import torch.nn as nn


class ReLUFactory(AbstractActivationFactory):
    """Activation function factory: returns ReLU"""

    def __init__(self, inplace=False):
        self.inplace = inplace

    def get_activation(self):
        return nn.ReLU(inplace=self.inplace)