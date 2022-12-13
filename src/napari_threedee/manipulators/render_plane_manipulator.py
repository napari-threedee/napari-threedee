import napari.layers
from napari.utils.events.event import EventBlocker
from napari.utils.geometry import rotation_matrix_from_vectors_3d
import numpy as np
from napari_threedee.manipulators.base_manipulator import BaseManipulator


class RenderPlaneManipulator(BaseManipulator):
    """A manipulator for moving and orienting an image layer rendering plane."""

    def __init__(self, viewer, layer=None):
        super().__init__(viewer, layer, rotator_axes='yz', translator_axes='z')

    def set_layers(self, layers: napari.layers.Image):
        super().set_layers(layers)

    def _connect_events(self):
        self.layer.plane.events.position.connect(self._update_transform)
        self.layer.plane.events.normal.connect(self._update_transform)

    def _disconnect_events(self):
        self.layer.plane.events.position.disconnect(self._update_transform)
        self.layer.plane.events.normal.disconnect(self._update_transform)

    def _update_transform(self):
        # get the new transformation data
        self._initialize_transform()

        # redraw
        self._backend._on_transformation_changed()

    def _initialize_transform(self):
        self.origin = np.array(self.layer.plane.position)
        plane_normal = self.layer.plane.normal
        self.rotation_matrix = rotation_matrix_from_vectors_3d([1, 0, 0], plane_normal)

    def _while_dragging_translator(self):
        with self.layer.plane.events.position.blocker(self._update_transform):
            self.layer.plane.position = self.origin

    def _while_dragging_rotator(self):
        with self.layer.plane.events.normal.blocker(self._update_transform):
            self.layer.plane.normal = self.z_vector

