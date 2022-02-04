from typing import Optional

import numpy as np
from napari_threedee.manipulators.base_manipulator import BaseManipulator


class LayerManipulator(BaseManipulator):
    """LayerManipulator is a manipulator for translating a layer.

    Parameters
    ----------

    """

    def __init__(self, viewer, layer, translator_length=50, order=0):
        self._line_length = translator_length
        self._initial_translate = None

        self._centroid = np.array([0, 0, 0])

        self._initial_translator_normals = np.asarray(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]
        )

        super().__init__(viewer, layer, order)

    def _pre_drag(
            self,
            click_point: np.ndarray,
            selected_translator: Optional[int],
            selected_rotator: Optional[int]
    ):
        self._initial_translate = self._layer.translate

    def _while_dragging_translator(self, selected_translator: int, translation_vector: np.ndarray):
        new_translation = self._initial_translate + translation_vector
        self._layer.translate = np.squeeze(new_translation)

    def _post_drag(self):
        self._initial_translate = None
