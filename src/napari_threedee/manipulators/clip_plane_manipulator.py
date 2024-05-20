import napari.layers
from napari.utils.events.event import EventBlocker
from napari.utils.geometry import rotation_matrix_from_vectors_3d
from napari.layers.utils.plane import ClippingPlane
import numpy as np
from napari_threedee.manipulators.base_manipulator import BaseManipulator
from napari_threedee.utils.napari_utils import data_to_world_normal, world_to_data_normal


class ClippingPlaneManipulator(BaseManipulator):
    """A manipulator for moving and orienting a layer clipping plane."""

    def __init__(self, viewer, layer=None):
        super().__init__(viewer, layer, rotator_axes='xyz', translator_axes='z')


    def set_layers(self, layers: napari.layers.Layer):
        super().set_layers(layers)

    def _connect_events(self):
        self.layer.experimental_clipping_planes[0].events.position.connect(self._update_transform)
        self.layer.experimental_clipping_planes[0].events.normal.connect(self._update_transform)
        self.layer.events.visible.connect(self._on_visibility_change)
        self.layer.events.depiction.connect(self._on_depiction_change)
        self._viewer.layers.events.removed.connect(self._disable_and_remove)

    def _disconnect_events(self):
        self.layer.experimental_clipping_planes[0].events.position.disconnect(self._update_transform)
        self.layer.experimental_clipping_planes[0].events.normal.disconnect(self._update_transform)

    def _update_transform(self):
        # get the new transformation data
        self._initialize_transform()

        # redraw
        self._backend._on_transformation_changed()

    def _initialize_transform(self):
        origin_world = self.layer.data_to_world(self.layer.experimental_clipping_planes[0].position)
        self.origin = np.array(origin_world)
        plane_normal_data = self.layer.experimental_clipping_planes[0].normal
        plane_normal_world = data_to_world_normal(vector=plane_normal_data, layer=self.layer)
        manipulator_normal = -1 * plane_normal_world
        self.rotation_matrix = rotation_matrix_from_vectors_3d([1, 0, 0], manipulator_normal)


    def _while_dragging_translator(self):
        with self.layer.experimental_clipping_planes[0].events.position.blocker(self._update_transform):
            self.layer.experimental_clipping_planes[0].position = self.layer.world_to_data(self.origin)

    def _while_dragging_rotator(self):
        with self.layer.experimental_clipping_planes[0].events.normal.blocker(self._update_transform):
            z_vector_data = world_to_data_normal(vector=self.z_vector, layer=self.layer)
            self.layer.experimental_clipping_planes[0].normal = z_vector_data

    def _on_depiction_change(self):
        if self.layer.depiction == 'plane':
            self.enabled = True
        else:
            self.enabled = False
