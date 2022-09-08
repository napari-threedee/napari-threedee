from napari.utils.geometry import intersect_line_with_plane_3d
import numpy as np

from ..utils.geometry import signed_angle_between_vectors, rotation_matrix_around_vector_3d


class RotatorDragManager:

    def __init__(self, normal_vector: np.ndarray, axis_index: int):
        self.rotation_normal = normal_vector
        self.axis_index = axis_index
        self._initial_click_vector = None
        self._initial_rotation_matrix = None
        self._initial_translation = None
        self._view_direction = None

        self._center_on_click_plane = None

    def setup_drag(self, click_point: np.ndarray, view_direction: np.ndarray, translation: np.ndarray, rotation_matrix=np.ndarray):

        click_on_rotation_plane = intersect_line_with_plane_3d(
            line_position=click_point,
            line_direction=view_direction,
            plane_position=translation,
            plane_normal=self.rotation_normal,
        )

        self._initial_click_vector = np.squeeze(click_on_rotation_plane) - translation
        self._initial_rotation_matrix = rotation_matrix.copy()
        self._initial_translation = translation
        self._view_direction = view_direction

    def update_drag(self,
                    click_point,
                    ):
        click_on_rotation_plane = intersect_line_with_plane_3d(
            line_position=click_point,
            line_direction=self._view_direction,
            plane_position=self._initial_translation,
            plane_normal=self.rotation_normal,
        )
        click_vector = np.squeeze(click_on_rotation_plane) - self._initial_translation

        rotation_angle = signed_angle_between_vectors(self._initial_click_vector, click_vector, self.rotation_normal)
        rotation_matrix = rotation_matrix_around_vector_3d(rotation_angle, self.rotation_normal)

        # update the rotation matrix and call the _while_rotator_drag callback
        updated_rotation_matrix = np.dot(rotation_matrix, self._initial_rotation_matrix)

        return self._initial_translation, updated_rotation_matrix
