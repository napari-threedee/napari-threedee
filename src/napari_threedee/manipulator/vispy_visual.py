from .manipulator import ManipulatorModel
from .manipulator_visual_data import ManipulatorVisualData

import numpy as np

from vispy.scene import Line, Compound, Markers


class ManipulatorVisual(Compound):
    _default_origin_point_kwargs = {
        'pos': np.array([0, 0, 0]),
        'size': 10,
        'edge_width_rel': 0.2,
        'edge_color': np.array([0.5, 0.5, 0.5]),
        'face_color': np.array([1, 1, 1]),
        'symbol': 'disc',
        'scaling': False,
        'antialias': 1,
        'spherical': True,
    }

    def __init__(self, parent):
        super().__init__([Line(), Line(), Markers(), Markers(), Line(), Markers()], parent=parent)

        # set up the central axis visual
        self.origin_marker_visual.spherical = True

        # set up the rotator visual
        self.rotator_handle_visual.spherical = True
        self.rotator_handle_visual.scaling = True
        self.rotator_handle_visual.antialias = 0

        # set up the translator visual
        self.translator_handle_visual.spherical = True
        self.translator_handle_visual.scaling = True
        self.translator_handle_visual.antialias = 0

    @property
    def origin_marker_visual(self) -> Markers:
        return self._subvisuals[3]

    @property
    def central_axes_visual(self) -> Line:
        return self._subvisuals[0]

    @property
    def rotator_line_visual(self) -> Line:
        return self._subvisuals[1]

    @property
    def rotator_handle_visual(self) -> Markers:
        return self._subvisuals[2]

    @property
    def translator_line_visual(self) -> Line:
        return self._subvisuals[4]

    @property
    def translator_handle_visual(self) -> Markers:
        return self._subvisuals[5]

    @classmethod
    def from_manipulator_visual_data(cls, manipulator_visual_data: ManipulatorVisualData):
        visual = cls()
        visual.
        return

    @classmethod
    def from_manipulator(cls, manipulator: ManipulatorModel):
