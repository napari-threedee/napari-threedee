from typing import List

import numpy as np
from pydantic import BaseModel

from napari_threedee._backend.manipulator._interface import _Grabbable
from napari_threedee._backend.manipulator.axis_model import AxisSet, AxisModel


class Translator(BaseModel, _Grabbable):
    """Axis, offset from origin with a handle at the base."""
    axis: AxisModel
    length: float = 3
    distance_from_origin: float = 21  # distance of start point from origin

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
        return self.start_point

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
