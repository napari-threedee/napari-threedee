from typing import Tuple, Optional

import numpy as np
from psygnal import EventedModel
from pydantic import validator, Field

from .central_axis import CentralAxisSet
from .rotator import RotatorSet
from .translator import TranslatorSet


class ManipulatorModel(EventedModel):
    translators: Optional[TranslatorSet]
    rotators: Optional[RotatorSet]
    central_axes: Optional[CentralAxisSet]

    origin: Tuple[float, float, float] = Field(default_factory=lambda: np.zeros(3))
    rotation_matrix: np.ndarray = Field(default_factory=lambda: np.eye(3))

    selected_axis_id: Optional[int]

    class Config:
        arbitrary_types_allowed = True

    @validator('central_axes', pre=True)
    def central_axes_from_string(cls, v):
        if isinstance(v, str):
            v = CentralAxisSet.from_string(v)
        return v

    @validator('translators', pre=True)
    def translators_from_string(cls, v):
        if isinstance(v, str):
            v = TranslatorSet.from_string(v)
        return v

    @validator('rotators', pre=True)
    def rotators_from_string(cls, v):
        if isinstance(v, str):
            v = RotatorSet.from_string(v)
        return v

    @classmethod
    def from_strings(cls, translators: Optional[str], rotators: Optional[str]):
        if rotators is None:
            return cls(central_axes=translators, translators=translators, rotators=rotators)
        rotator_central_axes = ''.join(
            rotator.axis.perpendicular_axes for rotator in RotatorSet.from_string(rotators)
        )
        translator_central_axes = '' if translators is None else translators
        central_axes = rotator_central_axes + translator_central_axes
        return cls(central_axes=central_axes, translators=translators, rotators=rotators)