from ..utils.napari_utils import add_mouse_callback_safe, remove_mouse_callback_safe
from ..mouse_callbacks import add_point_on_plane

from functools import partial


class PlanePointAnnotator:
    def __init__(self, viewer, image_layer, points_layer):
        self.viewer = viewer
        self.image_layer = image_layer
        self.points_layer = points_layer

        self.mouse_callback = self.generate_mouse_callback()
        self.bind_callbacks()

    def generate_mouse_callback(self):
        mouse_callback = partial(
            add_point_on_plane,
            points_layer=self.points_layer,
            plane_layer=self.image_layer
        )
        return mouse_callback

    def bind_callbacks(self):
        add_mouse_callback_safe(
            self.viewer.mouse_drag_callbacks, self.mouse_callback
        )