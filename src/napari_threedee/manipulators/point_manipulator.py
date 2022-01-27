from typing import Optional

import numpy as np
from napari_threedee.manipulators.base_manipulator import BaseManipulator


class PointManipulator(BaseManipulator):

    def __init__(self, viewer, layer, order=0, translator_length=50, rotator_radius=5):
        self._layer = layer
        self.ensure_point_selected()
        self._centroid = self.active_point_position
        self._initial_translator_normals = np.asarray(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )

        self._initial_rotator_normals = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ]
        )
        super().__init__(
            viewer,
            layer,
            order=order,
            translator_length=translator_length,
            rotator_radius=rotator_radius
        )

    @property
    def active_point_index(self):
        return list(self._layer.selected_data)[0]

    @property
    def active_point_position(self):
        return self._layer.data[self.active_point_index]

    def ensure_point_selected(self):
        if len(self._layer.selected_data) == 0:
            self._layer.selected_data = [0]

    def _pre_drag(
            self,
            click_point: np.ndarray,
            selected_translator: Optional[int],
            selected_rotator: Optional[int]
    ):
        pass

    def _while_translator_drag(self, selected_translator: int, translation_vector: np.ndarray):
        self._layer._move([self.active_point_index], self.centroid)

    def _while_rotator_drag(self, selected_rotator: int, rotation_matrix: np.ndarray):
        # todo: store rotmat somewhere
        pass

    def _on_click_cleanup(self):
        pass