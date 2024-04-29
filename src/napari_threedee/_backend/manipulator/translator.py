from typing import List

import numpy as np
from pydantic import BaseModel

from napari_threedee._backend.manipulator._interface import _Grabbable
from napari_threedee._backend.manipulator.axis_model import AxisSet, AxisModel


class Translator(BaseModel, _Grabbable):
    """Axis, offset from origin with a handle at the base."""
    axis: AxisModel
    distance_from_origin: float = 20  # distance of start point from origin
    handle_size: float = 10

    @property
    def length(self) -> np.ndarray:
        """length of the extra segment beyond the radius"""
        return np.floor(self.distance_from_origin/7)+1
        
    @property
    def start_point(self) -> np.ndarray:
        return self.distance_from_origin * np.array(self.axis.vector)

    @property
    def end_point(self) -> np.ndarray:
        return (self.distance_from_origin + self.length) * np.array(self.axis.vector)

    @property
    def points(self) -> np.ndarray:
        return np.stack([self.start_point, self.end_point], axis=0)

    @property
    def handle_point(self) -> np.ndarray:
        return self.end_point

    @classmethod
    def from_string(cls, axis: str):
        return cls(axis=AxisModel.from_string(axis))


class TranslatorSet(List[Translator]):
    @classmethod
    def from_axis_set(cls, axes: AxisSet):
        return cls(Translator(axis=axis) for axis in axes)

    @classmethod
    def from_string(cls, axes: str):
        if axes == '':
            return None
        return cls.from_axis_set(AxisSet.from_string(axes))

    def __str__(self) -> str:
        return ''.join(translator.axis.name for translator in self)
