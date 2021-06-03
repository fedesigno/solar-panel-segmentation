from abc import abstractmethod

from abc import ABC
import numpy as np


class BaseLogger(ABC):

    @abstractmethod
    def step(self, iteration: int = None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_scalar(self, name: str, value: float, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_image(self, name: str, image: np.ndarray, **kwargs) -> None:
        raise NotImplementedError()
