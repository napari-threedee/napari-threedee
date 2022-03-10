from typing import Optional, Union

import napari
import numpy as np
from napari_threedee.manipulators.base_manipulator import BaseManipulator


class LayerManipulator(BaseManipulator):
    """LayerManipulator is a manipulator for translating a layer.

    Parameters
    ----------

    """

    def __init__(self, viewer, layer=None, translator_length=50, order=0):
        super().__init__(
            viewer=viewer,
            layer=layer,
            translator_length=translator_length,
            order=order
        )

    def set_layers(self, layer: napari.layers.Layer):
        super().set_layers(layer)

    def _initialize_transform(self):
        self._translation = np.array([0, 0, 0])
        self._rot_mat = np.eye(3)

    def _set_initial_translation_vectors(self):
        self._initial_translation_vectors_ = np.asarray(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]
        )

    def _setup_translator_drag(self, click_point: np.ndarray, selected_translator: Optional[int]):
        if selected_translator is not None:
            self._initial_translation = np.array(self.layer.translate)

    def _process_translator_drag(
            self,
            event,
            selected_translator: int,
            initial_position_world: np.ndarray,
            selected_translator_normal: np.ndarray
    ):
        if selected_translator is None:
            # no processing necessary if a translator was not selected
            return
        # get drag vector projected onto the translator axis
        projected_distance = self.layer.projected_distance_from_mouse_drag(
            start_position=initial_position_world,
            end_position=event.position,
            view_direction=event.view_direction,
            vector=selected_translator_normal,
            dims_displayed=event.dims_displayed
        )
        translator_drag_vector = projected_distance * selected_translator_normal
        self.layer.translate = self._initial_translation + translator_drag_vector
        self._while_dragging_translator(selected_translator=selected_translator,
                                        translation_vector=translator_drag_vector)
