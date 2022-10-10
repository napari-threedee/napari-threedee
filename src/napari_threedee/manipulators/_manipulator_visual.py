import numpy as np

from vispy.scene import Line, Compound, Markers


class ManipulatorVisual(Compound):
    def __init__(self, parent):
        super().__init__([Line(), Line(), Markers(), Markers(), Line(), Markers()], parent=parent)

        # set up the central axis visual
        self.centroid_visual.spherical = True

        # set up the rotator visual
        self.rotator_handles_visual.spherical = True
        self.rotator_handles_visual.scaling = True
        self.rotator_handles_visual.antialias = 0

        # set up the translator visual
        self.translator_handles_visual.spherical = True
        self.translator_handles_visual.scaling = True
        self.translator_handles_visual.antialias = 0

    @property
    def axes_visual(self) -> Line:
        return self._subvisuals[0]

    @property
    def rotator_arc_visual(self) -> Line:
        return self._subvisuals[1]

    @property
    def rotator_handles_visual(self) -> Markers:
        return self._subvisuals[2]

    @property
    def centroid_visual(self) -> Markers:
        return self._subvisuals[3]

    @property
    def translator_visual(self) -> Line:
        return self._subvisuals[4]

    @property
    def translator_handles_visual(self) -> Markers:
        return self._subvisuals[5]

