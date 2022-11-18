import napari
import numpy as np

from napari_threedee.manipulators.base_manipulator import BaseManipulator


class LayerManipulator(BaseManipulator):
    """A manipulator for translating a layer."""

    def __init__(self, viewer, layer=None):
        super().__init__(viewer, layer, rotator_axes=None, translator_axes='xyz')

    def set_layers(self, layer: napari.layers.Layer):
        super().set_layers(layer)

    def _initialize_transform(self):
        self.origin = self.layer.translate[self.layer._dims_displayed]

    def _pre_drag(self):
        self.translate_start = self.layer.translate[self.layer._dims_displayed].copy()

    def _while_dragging_translator(self):
        new_translate = self.translate_start + self.origin
        self.layer.translate = new_translate
        # origin is relative to the layer transform so needs
        # to be reset after updating the transform
        self.origin = np.asarray((0, 0, 0))

