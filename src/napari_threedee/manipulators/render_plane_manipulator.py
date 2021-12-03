from typing import Optional

import numpy as np
from napari.utils.geometry import project_points_onto_plane, rotation_matrix_from_vectors
from napari_threedee.manipulators.base_manipulator import BaseManipulator


class RenderPlaneManipulator(BaseManipulator):

    def __init__(self, viewer, layer, order=0, line_length=50, rotator_radius=5):

        self._initial_plane_pos = None

        self._rotator_angle_offset = 0
        self._centroid = layer.experimental_slicing_plane.position
        normal = layer.experimental_slicing_plane.normal
        self._initial_translator_normals = np.asarray([normal])

        self._initial_rotator_normals = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ]
        )
        self._initial_rotator_normals[0] = layer.experimental_slicing_plane.normal

        super().__init__(
            viewer,
            layer,
            order=order,
            line_length=line_length,
            rotator_radius=rotator_radius
        )

    def _pre_drag(
            self,
            click_point: np.ndarray,
            selected_translator: Optional[int],
            selected_rotator: Optional[int]
    ):
        self._initial_plane_pos = self._layer.experimental_slicing_plane.position
        self._initial_rot_mat = self.rot_mat.copy()

    def _while_translator_drag(self, selected_translator: int, translation_vector: np.ndarray):
        new_translation = self._initial_plane_pos + translation_vector
        self._layer.experimental_slicing_plane.position = np.squeeze(new_translation)
        self.centroid = self._layer.experimental_slicing_plane.position

    def _while_rotator_drag(self, selected_rotator: int, rotation_matrix: np.ndarray):
        self.rot_mat = np.dot(rotation_matrix, self._initial_rot_mat)
        self._layer.experimental_slicing_plane.normal = self.rotator_normals[0]

    def _on_click_cleanup(self):
        self._initial_plane_pos = None
        self._initial_rot_mat = None
