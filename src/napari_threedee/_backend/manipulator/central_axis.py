from typing import List

import numpy as np
from pydantic import BaseModel

from napari_threedee._backend.manipulator.axis_model import AxisModel, AxisSet


class CentralAxis(BaseModel):
    axis: AxisModel
    length: float = 20

    @property
    def points(self) -> np.ndarray:
        return self.axis.points * self.length

    @classmethod
    def from_string(cls, axis: str):
        return cls(axis=AxisModel.from_string(axis))


class CentralAxisSet(List[CentralAxis]):
    @classmethod
    def from_axis_set(cls, axis_set: AxisSet):
        return cls(CentralAxis(axis=axis) for axis in axis_set)

    @classmethod
    def from_string(cls, axes: str):
        return cls.from_axis_set(AxisSet.from_string(axes))

    def __str__(self) -> str:
        return ''.join(central_axis.axis.name for central_axis in self)