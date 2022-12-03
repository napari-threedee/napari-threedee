from typing import Optional

import napari
from napari.layers import Image, Points

from napari_threedee._backend.threedee_model import ThreeDeeModel
from ..mouse_callbacks import add_point_on_plane
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, remove_mouse_callback_safe


class PlanePointAnnotator(ThreeDeeModel):
    def __init__(
            self,
            viewer: napari.Viewer,
            image_layer: Optional[Image] = None,
            points_layer: Optional[Points] = None,
            enabled: bool = False
    ):
        self.viewer = viewer
        self.image_layer = image_layer
        self.points_layer = points_layer
        if points_layer is None and image_layer is not None:
            self.points_layer = Points(data=[], ndim=image_layer.data.ndim)
            self.viewer.add_layer(self.points_layer)
        self.enabled = enabled

    def _mouse_callback(self, viewer, event):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        add_point_on_plane(
            viewer=viewer,
            event=event,
            points_layer=self.points_layer,
            plane_layer=self.image_layer
        )

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
