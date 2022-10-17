from typing import Tuple

import numpy as np
from psygnal import EventedModel
from pydantic import validator

from .central_axis import CentralAxisSet
from .rotator import RotatorSet
from .translator import TranslatorSet


class ManipulatorModel(EventedModel):
    # manipulable bits
    central_axes: CentralAxisSet
    rotators: RotatorSet
    translators: TranslatorSet

    # transformation
    origin: Tuple[float, float, float] = (0, 0, 0)
    rotation_matrix: np.ndarray = ((1, 0, 0),
                                   (0, 1, 0),
                                   (0, 0, 1))  # equality check failing when array is default

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
    def from_strings(cls, translators: str, rotators: str):
        translator_central_axes = translators
        rotators = RotatorSet.from_string(rotators)
        rotator_central_axes = ''.join(rotator.axis.perpendicular_axes for rotator in rotators)
        central_axes = ''.join(set(translator_central_axes + rotator_central_axes))
        return cls(central_axes=central_axes, translators=translators, rotators=rotators)
