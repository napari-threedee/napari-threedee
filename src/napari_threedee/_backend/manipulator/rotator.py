from typing import List

from pydantic import BaseModel
import numpy as np

from ._interface import _Grabbable
from .axis_model import AxisSet, AxisModel


class Rotator(BaseModel, _Grabbable):
    axis: AxisModel
    distance_from_origin: float = 20

    @property
    def handle_point(self):
        perpendicular_axes = AxisSet.from_string(self.axis.perpendicular_axes)
        handle_vector = perpendicular_axes[0].vector + perpendicular_axes[1].vector
        normalised_handle_vector = handle_vector / np.linalg.norm(handle_vector)
        return normalised_handle_vector * self.distance_from_origin

    @classmethod
    def from_string(cls, axis: str):
        return cls(axis=AxisModel.from_string(axis))


class RotatorSet(List[Rotator]):
    @classmethod
    def from_axis_set(cls, axes: AxisSet):
        return cls(Rotator(axis=axis) for axis in axes)

    @classmethod
    def from_string(cls, axes: str):
        if axes == '':
            return None
        return cls.from_axis_set(AxisSet.from_string(axes))
