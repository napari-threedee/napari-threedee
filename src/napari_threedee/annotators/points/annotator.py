from typing import Optional

import napari.layers
import napari.types
import numpy as np

from napari_threedee._backend.threedee_model import N3dComponent
from napari_threedee.manipulators.constants import ADD_POINT_KEY
from napari_threedee.utils.mouse_callbacks import on_mouse_alt_click_add_point_on_plane
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, \
    remove_mouse_callback_safe, add_point_on_plane


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

    def _add_point_on_mouse_alt_click(self, layer, event):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        on_mouse_alt_click_add_point_on_plane(
            viewer=self.viewer,
            event=event,
            points_layer=self.points_layer,
            image_layer=self.image_layer
        )

    def _add_point_on_key_press(self, *args):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        add_point_on_plane(
            viewer=self.viewer,
            image_layer=self.image_layer,
            points_layer=self.points_layer,
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
        if self.image_layer is not None:
            add_mouse_callback_safe(
                self.image_layer.mouse_drag_callbacks, self._add_point_on_mouse_alt_click, index=0
            )
            self.image_layer.bind_key(ADD_POINT_KEY, self._add_point_on_key_press, overwrite=True)

    def _on_disable(self):
        if self.image_layer is not None:
            remove_mouse_callback_safe(
                self.image_layer.mouse_drag_callbacks, self._add_point_on_mouse_alt_click
            )
        if self.image_layer is not None:
            self.image_layer.bind_key(ADD_POINT_KEY, None, overwrite=True)
