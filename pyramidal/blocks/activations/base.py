from abc import ABC, abstractmethod


class AbstractActivationFactory(ABC):
    """Abstract base class for activation functions factories"""

    @abstractmethod
    def get_activation(self):
        pass
