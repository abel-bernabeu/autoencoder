from abc import ABC, abstractmethod


class AbstractDownFactory(ABC):
    """Abstract base class for downsamplers factories"""

    @abstractmethod
    def get_down(self, channels):
        pass