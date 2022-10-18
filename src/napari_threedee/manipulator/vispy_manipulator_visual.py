from .model import ManipulatorModel
from .vispy_visual_data import ManipulatorVisualData

import numpy as np

from vispy.scene import Line, Compound, Markers


class ManipulatorVisual(Compound):
    def __init__(self, parent, manipulator_visual_data: ManipulatorVisualData):
        super().__init__([Line(), Line(), Markers(), Markers(), Line(), Markers()], parent=parent)
        self.unfreeze()
        self._manipulator_visual_data = manipulator_visual_data
        self.freeze()

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

    @classmethod
    def from_manipulator(cls, manipulator: ManipulatorModel):
        mvd = ManipulatorVisualData.from_manipulator(manipulator)
        return cls(parent=None, manipulator_visual_data=mvd)

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

    def update_visuals_from_manipulator_visual_data(self):
        self._update_central_axis_visuals()
        self._update_translator_visuals()
        self._update_rotator_visuals()
        self.update()

    def _setup_origin_marker_visual(self):
        self.origin_marker_visual.set_data(
            pos=np.array([[0, 0, 0]]),
            face_color=[0.7, 0.7, 0.7, 1],
            size=10
        )

    def _update_central_axis_visuals(self):
        self.central_axes_visual.set_data(
            pos=self.manipulator_visual_data.central_axis_line_data.vertices,
            connect=self.manipulator_visual_data.central_axis_line_data.connections,
            color=self.manipulator_visual_data.central_axis_line_colors,
            width=self.manipulator_visual_data.central_axis_line_data.line_width,
        )

    def _update_translator_visuals(self):
        self.translator_line_visual.set_data(
            pos=self.manipulator_visual_data.translator_line_data.vertices,
            connect=self.manipulator_visual_data.translator_line_data.connections,
            color=self.manipulator_visual_data.translator_line_colors,
            width=self.manipulator_visual_data.translator_line_data.line_width,
        )
        self.translator_handle_visual.set_data(
            pos=self.manipulator_visual_data.translator_handle_data.points,
            face_color=self.manipulator_visual_data.translator_handle_data.colors,
            size=self.manipulator_visual_data.translator_handle_data.handle_size,
        )

    def _update_rotator_visuals(self):
        self.rotator_line_visual.set_data(
            pos=self.manipulator_visual_data.rotator_line_data.vertices,
            connect=self.manipulator_visual_data.rotator_line_data.connections,
            color=self.manipulator_visual_data.rotator_line_colors,
            width=self.manipulator_visual_data.rotator_line_data.line_width,
        )
        self.rotator_handle_visual.set_data(
            pos=self.manipulator_visual_data.rotator_handle_data.points,
            face_color=self.manipulator_visual_data.rotator_handle_data.colors,
            size=self.manipulator_visual_data.rotator_handle_data.handle_size,
        )
