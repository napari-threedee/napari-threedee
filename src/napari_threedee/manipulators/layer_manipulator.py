import napari
import numpy as np

from napari_threedee.manipulators.base_manipulator import BaseManipulator
from napari_threedee.utils.napari_utils import get_dims_displayed


class LayerManipulator(BaseManipulator):
    """A manipulator for translating a layer."""

    def __init__(self, viewer, layer=None):
        super().__init__(viewer, layer, rotator_axes=None, translator_axes='xyz')

    def set_layers(self, layer: napari.layers.Layer):
        super().set_layers(layer)

    def _connect_events(self):
        self.layer.events.visible.connect(self._on_visibility_change)
        self._viewer.layers.events.removed.connect(self._disable_and_remove)

    def _initialize_transform(self):
        self.origin = np.asarray(self.layer.data_to_world((0, 0, 0)))
        print(self.origin)

    def _pre_drag(self):
        self.translate_start = self.origin.copy()

    def _while_dragging_translator(self):
        self.layer.translate = self.origin
