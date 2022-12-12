from typing import List

import numpy as np
import napari

from napari_threedee._backend.threedee_model import ThreeDeeModel
from napari_threedee.utils.napari_utils import get_napari_visual


class LightingControl(ThreeDeeModel):
    def __init__(self, viewer: napari.Viewer):
        self._viewer = viewer
        self._selected_layers = []
        self._selected_layer_visuals = []
        self.enabled = False

    def set_layers(self, layers: List[napari.layers.Surface]):
        self.selected_layers = layers
        self._on_camera_change()

    def _on_enable(self):
        self._connect_events()
        self._enabled = True
        self._on_camera_change()
        for layer in self.selected_layers:
            layer.events.shading()

    def _on_disable(self):
        self._disconnect_events()

    @property
    def selected_layers(self) -> List[napari.layers.Surface]:
        return self._selected_layers

    @selected_layers.setter
    def selected_layers(self, layers: List[napari.layers.Surface]):
        if isinstance(layers, napari.layers.base.Layer):
            layers = [layers]
        self._selected_layer_visuals = [
            get_napari_visual(viewer=self._viewer, layer=layer) for layer in layers
        ]
        self._selected_layers = layers

    @property
    def selected_layer_visuals(self):
        return self._selected_layer_visuals

    def _on_camera_change(self, event=None):
        if self.enabled is False:
            # only update lighting direction if enabled
            return
        view_direction = np.asarray(self._viewer.camera.view_direction)

        for layer, visual in zip(self.selected_layers, self.selected_layer_visuals):
            dims_displayed = layer._dims_displayed
            layer_view_direction = np.asarray(layer._world_to_data_ray(view_direction))[dims_displayed]
            visual.node.shading_filter.light_dir = layer_view_direction[::-1]

    def _connect_events(self):
        self._viewer.camera.events.angles.connect(self._on_camera_change)

    def _disconnect_events(self):
        self._viewer.camera.events.angles.disconnect(self._on_camera_change)
