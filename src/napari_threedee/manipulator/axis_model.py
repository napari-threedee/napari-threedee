from typing import Tuple, List

import numpy as np
from pydantic import BaseModel, validator


class AxisModel(BaseModel):
    name: str
    vector: np.ndarray  # (3, ) array
    perpendicular_axes: str
    color: np.ndarray  # (4, ) array
    id: int

    class Config:
        arbitrary_types_allowed = True

    @validator('name')
    def lowercase(cls, v: str):
        return v.lower()

    @property
    def points(self):
        """(2, 3) array of points."""
        return np.stack([(0, 0, 0), self.vector], axis=0)

    @classmethod
    def from_string(cls, axis: str):
        axis = axis.lower()
        if axis not in 'xyz':
            raise ValueError("axis must be 'x' 'y' or 'z'.")
        if axis == 'x':
            return cls(
                name='x',
                vector=np.array([0, 0, 1]),
                perpendicular_axes='yz',
                color=np.array([1, 0.75, 0.52, 1]),
                id=2,
            )
        elif axis == 'y':
            return cls(
                name='y',
                vector=np.array([0, 1, 0]),
                perpendicular_axes='xz',
                color=np.array([0.75, 0.68, 0.83, 1]),
                id=1,
            )
        elif axis == 'z':
            return cls(
                name='z',
                vector=np.array([1, 0, 0]),
                perpendicular_axes='xy',
                color=np.array([0.5, 0.8, 0.5, 1]),
                id=0
            )

    @classmethod
    def from_id(cls, id: int):
        if id == 0:
            return cls.from_string('z')
        elif id == 1:
            return cls.from_string('y')
        elif id == 2:
            return cls.from_string('x')
        else:
            raise ValueError("id must be 0, 1 or 2")


class AxisSet(List[AxisModel]):
    @classmethod
    def from_string(cls, axes: str):
        axes = ''.join(set(axes))
        return cls(AxisModel.from_string(character) for character in axes)

    @property
    def perpendicular_axes(self):
        """AxisSet containing axes perpendicular to those in this set of axes."""
        return self.from_string(''.join(axis.perpendicular_axes for axis in self))
