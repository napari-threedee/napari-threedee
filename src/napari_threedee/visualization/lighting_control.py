from typing import List

import igl
import numpy as np
import napari

from napari_threedee._backend.threedee_model import ThreeDeeModel
from napari_threedee.utils.napari_utils import (
    get_dims_displayed,
    get_napari_visual,
)


class LightingControl(ThreeDeeModel):
    def __init__(self, viewer: napari.Viewer):
        self._viewer = viewer
        self._selected_layers = []
        self._selected_layer_visuals = []
        self._ambient_occlusion = False
        self._pre_ambient_occlusion_values = []
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
    def ambient_occlusion(self) -> bool:
        return self._ambient_occlusion

    @ambient_occlusion.setter
    def ambient_occlusion(self, value: bool) -> None:
        if value == self.ambient_occlusion:
            # if the value isn't changed, do nothing
            return
        self._ambient_occlusion = value

        if value is True:
            self._activate_ambient_occlusion()
        else:
            self._deactivate_ambient_occlusion()

    def _activate_ambient_occlusion(self):
        self._compute_ambient_occlusion()

    def _compute_ambient_occlusion(self):
        self._pre_ambient_occlusion_values = []
        for layer, visual in zip(self._selected_layers, self._selected_layer_visuals):
            vertices, faces, vertex_values = layer.data
            self._pre_ambient_occlusion_values.append(vertex_values)
            vertex_normals = igl.per_vertex_normals(vertices, faces)
            ao = igl.ambient_occlusion(vertices, faces, vertices, vertex_normals, 20)
            attenuation_factor = 1 - ao

            #
            attenuated_values = vertex_values * attenuation_factor

            # set the data
            # meshdata = visual.node._meshdata
            # meshdata.set_vertex_values(attenuated_values)
            # visual.node.set_data(meshdata=meshdata)

    def _calc(self, vertices, faces, vertex_values):
        vertex_normals = igl.per_vertex_normals(vertices, faces)
        ao = igl.ambient_occlusion(vertices, faces, vertices, vertex_normals, 20)
        attenuation_factor = 1 - ao
    def _deactivate_ambient_occlusion(self):
        # restore the values
        for values, layer, visual in zip(
                self._pre_ambient_occlusion_values,
                self._selected_layers,
                self._selected_layer_visuals,
        ):
            # set the data
            meshdata = visual.node._meshdata
            meshdata.set_vertex_values(values)
            visual.node.set_data(meshdata=meshdata)

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
            dims_displayed = get_dims_displayed(layer)
            layer_view_direction = np.asarray(layer._world_to_data_ray(view_direction))[dims_displayed]
            visual.node.shading_filter.light_dir = layer_view_direction[::-1]

    def _connect_events(self):
        self._viewer.camera.events.angles.connect(self._on_camera_change)

    def _disconnect_events(self):
        self._viewer.camera.events.angles.disconnect(self._on_camera_change)
