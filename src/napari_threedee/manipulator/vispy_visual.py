from .manipulator import ManipulatorModel
from .manipulator_visual_data import ManipulatorVisualData

import numpy as np

from vispy.scene import Line, Compound, Markers


class ManipulatorVisual(Compound):
    def __init__(self, parent, manipulator_visual_data: ManipulatorVisualData):
        super().__init__([Line(), Line(), Markers(), Markers(), Line(), Markers()], parent=parent)
        self._manipulator_visual_data = manipulator_visual_data

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
    def manipulator_visual_data(self) -> ManipulatorVisualData:
        return self._manipulator_visual_data

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

    def _instantiate_from_manipulator_visual_data(self):
        self._setup_central_axis_visuals()
        self._setup_translator_visuals()
        self._setup_rotator_visuals()

    def _setup_central_axis_visuals(self):
        self.central_axes_visual.set_data(
            pos=self.manipulator_visual_data.central_axis_line_data.vertices,
            connect=self.manipulator_visual_data.central_axis_line_data.connections,
            color=self.manipulator_visual_data.central_axis_line_data.colors,
            width=self.manipulator_visual_data.central_axis_line_data.line_width,
        )

    def _setup_translator_visuals(self):
        self.translator_line_visual.set_data(
            pos=self.manipulator_visual_data.translator_line_data.vertices,
            connect=self.manipulator_visual_data.translator_line_data.connections,
            color=self.manipulator_visual_data.translator_line_data.colors,
            width=self.manipulator_visual_data.translator_line_data.line_width,
        )

    def _setup_rotator_visuals(self):
        self.rotator_line_visual.set_data(
            pos=self.manipulator_visual_data.rotator_line_data.vertices,
            connect=self.manipulator_visual_data.rotator_line_data.connections,
            color=self.manipulator_visual_data.rotator_line_data.colors,
            width=self.manipulator_visual_data.rotator_line_data.line_width,
        )

    @classmethod
    def from_manipulator_visual_data(cls, manipulator_visual_data: ManipulatorVisualData):
        visual = cls(manipulator_visual_data=manipulator_visual_data)
        visual._instantiate_from_manipulator_visual_data()
        return visual

    @classmethod
    def from_manipulator(cls, manipulator: ManipulatorModel):
        mvd = ManipulatorVisualData.from_manipulator(manipulator)
        return cls.from_manipulator_visual_data(mvd)

