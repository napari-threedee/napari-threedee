import napari.layers
from napari.utils.events.event import EventBlocker
from napari.utils.geometry import rotation_matrix_from_vectors_3d
import numpy as np
from napari_threedee.manipulators.base_manipulator import BaseManipulator
from napari_threedee.utils.napari_utils import data_to_world_normal, world_to_data_normal


class RenderPlaneManipulator(BaseManipulator):
    """A manipulator for moving and orienting an image layer rendering plane."""

    def __init__(self, viewer, layer=None):
        super().__init__(viewer, layer, rotator_axes='xyz', translator_axes='z')

    def set_layers(self, layers: napari.layers.Image):
        super().set_layers(layers)

    def _connect_events(self):
        self.layer.plane.events.position.connect(self._update_transform)
        self.layer.plane.events.normal.connect(self._update_transform)
        self.layer.events.visible.connect(self._on_visibility_change)
        self.layer.events.depiction.connect(self._on_depiction_change)
        self._viewer.layers.events.removed.connect(self._disable_and_remove)
        self._viewer.camera.events.connect(self._update_transform)

    def _disconnect_events(self):
        self.layer.plane.events.position.disconnect(self._update_transform)
        self.layer.plane.events.normal.disconnect(self._update_transform)
        self._viewer.camera.events.disconnect(self._update_transform)

    def _update_transform(self):
        # ensure the manipulator is clamped to the layer extent
        self._backend.clamp_to_layer_bbox = True
        # get the new transformation data
        self._initialize_transform()

        # redraw
        self._backend._on_transformation_changed()

    def _initialize_transform(self):
        origin_world = self.layer.data_to_world(self.layer.plane.position)
        self.origin = np.array(origin_world)
        plane_normal_data = self.layer.plane.normal
        plane_normal_world = data_to_world_normal(vector=plane_normal_data, layer=self.layer)
        manipulator_normal = plane_normal_world

        # flip the manipulator so it's always visible
        if np.dot(manipulator_normal, self._viewer.camera.up_direction) < 0:
            manipulator_normal *= -1
        self.rotation_matrix = rotation_matrix_from_vectors_3d([1, 0, 0], manipulator_normal)


    def _while_dragging_translator(self):
        with self.layer.plane.events.position.blocker(self._update_transform):
            self.layer.plane.position = self.layer.world_to_data(self.origin)

    def _while_dragging_rotator(self):
        with self.layer.plane.events.normal.blocker(self._update_transform):
            z_vector_data = world_to_data_normal(vector=self.z_vector, layer=self.layer)
            self.layer.plane.normal = z_vector_data

    def _on_depiction_change(self):
        if self.layer.depiction == 'plane':
            self.enabled = True
        else:
            self.enabled = False
