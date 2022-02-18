from functools import cached_property
from typing import Optional

import napari.layers
import numpy as np
from napari_threedee.manipulators.base_manipulator import BaseManipulator


class RenderPlaneManipulator(BaseManipulator):
    """A manipulator for moving an image layer rendering plane.
    """

    def __init__(self, viewer, layer=None, order=0, translator_length=50, rotator_radius=5):

        self.layer = layer
        if self.layer is not None:
            self._translation = np.array(self.layer.experimental_slicing_plane.position)
        else:
            self._translation = np.array([0, 0, 0])

        super().__init__(
            viewer,
            layer,
            order=order,
            translator_length=translator_length,
            rotator_radius=rotator_radius
        )

    def set_layers(self, layer: napari.layers.Image):
        super().set_layers(layer)

    @cached_property
    def _initial_translation_vectors(self):
        if self.layer is not None:
            return np.asarray([self.layer.experimental_slicing_plane.normal])
        else:
            return super()._initial_translation_vectors

    @cached_property
    def _initial_rotator_normals(self):
        # if self.layer is None:
        #     return super()._initial_rotator_normals
        normals = np.array(
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]
                ]
            )
        normals[0] = self.layer.experimental_slicing_plane.normal
        return normals

    def _pre_drag(
            self,
            click_point: np.ndarray,
            selected_translator: Optional[int],
            selected_rotator: Optional[int]
    ):
        self._initial_plane_pos = np.asarray(self._layer.experimental_slicing_plane.position)
        self._initial_rot_mat = self.rot_mat.copy()

    def _while_dragging_translator(self, selected_translator: int, translation_vector: np.ndarray):
        new_translation = self._initial_plane_pos + translation_vector
        self._layer.experimental_slicing_plane.position = np.squeeze(new_translation)
        self.centroid = np.asarray(self._layer.experimental_slicing_plane.position)

    def _while_dragging_rotator(self, selected_rotator: int, rotation_matrix: np.ndarray):
        self._layer.experimental_slicing_plane.normal = self.rotator_normals[0]

    def _post_drag(self):
        self._initial_plane_pos = None
