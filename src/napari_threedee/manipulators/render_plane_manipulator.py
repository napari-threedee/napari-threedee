import napari.layers
from napari.utils.geometry import rotation_matrix_from_vectors_3d
import numpy as np
from napari_threedee.manipulators.base_manipulator import BaseManipulator


class RenderPlaneManipulator(BaseManipulator):
    """A manipulator for moving and orienting an image layer rendering plane."""

    def __init__(self, viewer, layer=None):
        super().__init__(viewer, layer, rotator_axes='yz', translator_axes='z')

    def set_layers(self, layers: napari.layers.Image):
        super().set_layers(layers)

    def _initialize_transform(self):
        self.origin = np.array(self.layer.plane.position)
        plane_normal = self.layer.plane.normal
        self.rotation_matrix = rotation_matrix_from_vectors_3d([1, 0, 0], plane_normal)

    def _while_dragging_translator(self):
        self.layer.plane.position = self.origin

    def _while_dragging_rotator(self):
        self.layer.plane.normal = self.z_vector
