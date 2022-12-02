import napari
from napari.utils.geometry import intersect_line_with_plane_3d
import numpy as np

from ...utils.geometry import signed_angle_between_vectors, rotation_matrix_around_vector_3d
from ...utils.napari_utils import get_mouse_position_in_displayed_layer_data_coordinates


class RotatorDragManager:

    def __init__(self, rotation_vector: np.ndarray):
        self.rotation_vector = rotation_vector
        self._layer = None
        self._initial_click_vector = None
        self._initial_rotation_matrix = None
        self._initial_translation = None
        self._view_direction = None

        self._center_on_click_plane = None

    def setup_drag(self, layer: napari.layers.Layer, mouse_event, translation: np.ndarray,
                   rotation_matrix=np.ndarray):
        self._layer = layer

        click_point_data, click_dir_data_3d = get_mouse_position_in_displayed_layer_data_coordinates(layer,
                                                                                                     mouse_event)
        click_on_rotation_plane = intersect_line_with_plane_3d(
            line_position=click_point_data,
            line_direction=click_dir_data_3d,
            plane_position=translation,
            plane_normal=self.rotation_vector,
        )

        self._initial_click_vector = np.squeeze(click_on_rotation_plane) - translation
        self._initial_rotation_matrix = rotation_matrix.copy()
        self._initial_translation = translation
        self._view_direction = click_dir_data_3d

    def update_drag(self, mouse_event):
        click_point_data = np.asarray(self._layer.world_to_data(mouse_event.position))[
            mouse_event.dims_displayed]
        click_on_rotation_plane = intersect_line_with_plane_3d(
            line_position=click_point_data,
            line_direction=self._view_direction,
            plane_position=self._initial_translation,
            plane_normal=self.rotation_vector,
        )
        click_vector = np.squeeze(click_on_rotation_plane) - self._initial_translation
        rotation_angle = signed_angle_between_vectors(
            self._initial_click_vector, click_vector, self.rotation_vector
        )
        rotation_matrix = rotation_matrix_around_vector_3d(rotation_angle, self.rotation_vector)

        # update the rotation matrix and call the _while_rotator_drag callback
        updated_rotation_matrix = np.dot(rotation_matrix, self._initial_rotation_matrix)

        return self._initial_translation, updated_rotation_matrix


class TranslatorDragManager:
    def __init__(self, translation_vector: np.ndarray):
        self.translation_vector = translation_vector

        self._initial_rotation_matrix = None
        self._initial_translation = None
        self._view_direction = None
        self._initial_position_world = None

    def setup_drag(self, layer: napari.layers.Layer, mouse_event, translation: np.ndarray,
                   rotation_matrix=np.ndarray):
        self._layer = layer

        _, click_dir_data_3d = get_mouse_position_in_displayed_layer_data_coordinates(layer, mouse_event)

        self._initial_position_world = mouse_event.position
        self._initial_rotation_matrix = np.copy(rotation_matrix)
        self._initial_translation = np.copy(translation)
        self._view_direction = click_dir_data_3d

    def update_drag(self, mouse_event):
        projected_distance = self._layer.projected_distance_from_mouse_drag(
            start_position=self._initial_position_world,
            end_position=mouse_event.position,
            view_direction=mouse_event.view_direction,
            vector=self.translation_vector,
            dims_displayed=mouse_event.dims_displayed
        )
        translator_drag_vector = projected_distance * self.translation_vector
        updated_translation = self._initial_translation + translator_drag_vector

        return updated_translation, self._initial_rotation_matrix
