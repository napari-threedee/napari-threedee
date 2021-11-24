import numpy as np
from napari_threedee.manipulators.base_manipulator import BaseManipulator


class LayerManipulator(BaseManipulator):

    def __init__(self, viewer, layer, line_length=50, order=0):
        self._line_length = line_length
        self._initial_translate = None

        super().__init__(viewer, layer, order)

    @property
    def line_length(self):
        return self._line_length

    @line_length.setter
    def line_length(self, line_length):
        self._line_length = line_length
        self._on_data_change()


    def _init_arrow_lines(self):
        # note order is x, y, z for VisPy
        centroid = np.mean(self._layer._extent_data, axis=0)
        self._line_data2D = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]]
        )
        line_data3D = self.line_length * np.array(
            [[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]]
        )
        self._line_data3D = line_data3D + centroid

    def _pre_drag(self, click_point_data_displayed, rotator_index):
        self._initial_translate = self._layer.translate

    def _while_translator_drag(self, translation_vector: np.ndarray, rotator_drag_vector: np.ndarray, rotator_selection: np.ndarray):
        new_translation = self._initial_translate + translation_vector
        self._layer.translate = np.squeeze(new_translation)

    def _while_rotator_drag(self, click_position, rotation_drag_vector, rotator_selection):
        pass

    def _on_click_cleanup(self):
        self._initial_translate = None
