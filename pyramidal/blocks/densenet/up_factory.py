from abc import ABC, abstractmethod


class AbstractUpFactory(ABC):
    """Abstract base class for upsamplers factories"""

    @abstractmethod
    def get_up(self, channels):
        pass