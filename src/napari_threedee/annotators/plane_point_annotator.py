from functools import partial
from typing import Optional, TYPE_CHECKING

from ..utils.napari_utils import add_mouse_callback_safe, remove_mouse_callback_safe
from ..mouse_callbacks import add_point_on_plane

import napari
from napari.layers import Image, Points


class PlanePointAnnotator:
    def __init__(
            self,
            viewer: napari.Viewer,
            image_layer: Optional[Image] = None,
            points_layer: Optional[Points] = None,
            annotating: bool = False
    ):
        self.viewer = viewer
        self.image_layer = image_layer
        self.points_layer = points_layer
        self.annotating = annotating

    @property
    def annotating(self) -> bool:
        return self._annotating

    @annotating.setter
    def annotating(self, value: bool):
        if value is True:
            self.bind_callbacks()
        else:
            self.unbind_callbacks()
        self._annotating = value

    def _mouse_callback(self, viewer, event):
        if (self.image_layer is None) and (self.points_layer is None):
            return
        add_point_on_plane(
            viewer=viewer,
            event=event,
            points_layer=self.points_layer,
            plane_layer=self.image_layer
        )

    def bind_callbacks(self):
        add_mouse_callback_safe(
            self.viewer.mouse_drag_callbacks, self._mouse_callback
        )

    def unbind_callbacks(self):
        remove_mouse_callback_safe(
            self.viewer.mouse_drag_callbacks, self._mouse_callback
        )