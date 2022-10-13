from enum import Enum
from typing import Optional

import napari
from napari.layers import Image
import numpy as np

from ..base import ThreeDeeModel
from ..annotators.spline_annotator import SplineAnnotator


class CameraSplineMode(Enum):
    EXPLORE = "explore"
    ANNOTATE = "annotate"


class CameraSpline(ThreeDeeModel):
    COLOR_CYCLE = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
    ]
    SPLINE_ID_COLUMN: str = "spline_id"
    N_SPLINE_POINTS = 1000

    def __init__(
            self,
            viewer: napari.Viewer,
            image_layer: Optional[Image] = None,
            enabled: bool = False
    ):

        self.viewer = viewer
        self._image_layer = image_layer

        self.spline_annotator_model = SplineAnnotator(viewer=viewer, image_layer=None, enabled=False)

        self.enabled = enabled

    @property
    def mode(self) -> CameraSplineMode:
        return self._mode

    @mode.setter
    def mode(self, mode: CameraSplineMode):
        self._mode = mode

    @property
    def image_layer(self) -> Image:
        return self._image_layer

    @image_layer.setter
    def image_layer(self, layer: Image):
        self._image_layer = layer
        # self.spline_annotator_model.set_layers(layer)

    def start_spline_annotation(self):
        self.spline_annotator_model.enabled = True

        # disable the key binding to switch to the next spline index
        self.viewer.bind_key('n', None)

    def stop_spline_annotation(self):
        self.spline_annotator_model.enabled = False

    def _on_enable(self):
        self.spline_annotator_model.set_layers(self.image_layer)

    def _on_disable(self):
        self.spline_annotator_model.enabled = False

    def set_layers(self, image_layer: napari.layers.Image):
        self.image_layer = image_layer

    def set_camera_position(self, spline_coordinate: float):
        if self.image_layer is None:
            # do not do anything if the image layer hasn't been set
            return

        spline_dict = self.spline_annotator_model.points_layer.metadata["splines"]

        # only one spline
        spline_object = spline_dict[0]
        spline_point = spline_object._sample_backbone(u=[spline_coordinate])
        print(spline_point)
        self.viewer.camera.center = np.squeeze(spline_point)

        view_direction = np.squeeze(spline_object._sample_backbone(u=[spline_coordinate], derivative=1))
        view_direction_displayed = view_direction[list(self.viewer.dims.displayed)]
        self.viewer.camera.set_view_direction(
            view_direction=view_direction_displayed, up_direction=self.viewer.camera.up_direction
        )

