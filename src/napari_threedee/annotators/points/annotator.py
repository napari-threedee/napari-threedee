from typing import Optional

import napari.layers
import napari.types
import numpy as np

from napari_threedee._backend.threedee_model import N3dComponent
from napari_threedee.annotators.points.validation import validate_layer
from napari_threedee.utils.mouse_callbacks import add_point_on_plane
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, \
    remove_mouse_callback_safe


class PointAnnotator(N3dComponent):
    def __init__(
        self,
        viewer: napari.Viewer,
        image_layer: Optional[napari.layers.Image] = None,
        points_layer: Optional[napari.layers.Points] = None,
        enabled: bool = False
    ):
        self.viewer = viewer
        self.image_layer = image_layer
        if points_layer is None:
            points_layer = self._create_points_layer()
        self.points_layer = points_layer
        self.enabled = enabled

    def _mouse_callback(self, viewer, event):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        add_point_on_plane(
            viewer=viewer,
            event=event,
            points_layer=self.points_layer,
            image_layer=self.image_layer
        )

    def _create_points_layer(self) -> napari.layers.Points:
        from napari_threedee.data_models import N3dPoints
        ndim = self.image_layer.ndim if self.image_layer is not None else 3
        return N3dPoints(data=np.empty(shape=(0, ndim))).as_layer()

    def set_layers(
        self,
        image_layer: napari.layers.Image,
        points_layer: napari.layers.Points
    ):
        self.image_layer = image_layer
        self.points_layer = points_layer

    def _on_enable(self):
        add_mouse_callback_safe(
            self.viewer.mouse_drag_callbacks, self._mouse_callback
        )

    def _on_disable(self):
        remove_mouse_callback_safe(
            self.viewer.mouse_drag_callbacks, self._mouse_callback
        )
