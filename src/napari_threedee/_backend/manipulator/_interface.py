from abc import ABC, abstractmethod

import numpy as np


class _Grabbable(ABC):

    @property
    @abstractmethod
    def handle_point(self) -> np.ndarray:
        """(3, )"""
        ...
