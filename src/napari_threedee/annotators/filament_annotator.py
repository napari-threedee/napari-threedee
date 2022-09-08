from typing import Optional

import napari
from napari.layers import Image, Points

from .plane_point_annotator import PlanePointAnnotator
from ..utils.napari_utils import add_mouse_callback_safe, remove_mouse_callback_safe


class FilamentAnnotator(PlanePointAnnotator):
    COLOR_CYCLE = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
    ]
    FILAMENT_ID_LABEL = 'filament_id'

    def __init__(
            self,
            viewer: napari.Viewer,
            image_layer: Optional[Image] = None,
            enabled: bool = False
    ):
        self.viewer = viewer
        self.image_layer = image_layer
        self.points_layer = None
        self.enabled = enabled

        self.current_filament_id: int = 0
        self.viewer.bind_key('n', self.next_filament)

        if image_layer is not None:
            self.set_layers(self.image_layer)

    @property
    def current_filament_id(self):
        return self._current_filament_id

    @current_filament_id.setter
    def current_filament_id(self, id: int):
        self._current_filament_id = id
        if self.points_layer is not None:
            self.points_layer.current_properties = {
                self.FILAMENT_ID_LABEL: self.current_filament_id
            }

    def next_filament(self, event=None):
        self.current_filament_id += 1

    def previous_filament(self, event=None):
        self.current_filament_id -= 1

    def _mouse_callback(self, viewer, event):
        super()._mouse_callback(viewer, event)

    def _create_points_layer(self) -> Optional[Points]:
        layer = Points(
            data=[0] * self.image_layer.data.ndim,
            ndim=self.image_layer.data.ndim,
            name='filaments',
            features={self.FILAMENT_ID_LABEL: [0]},
            face_color=self.FILAMENT_ID_LABEL,
            face_color_cycle=self.COLOR_CYCLE)
        layer.selected_data = {0}
        layer.remove_selected()
        self.current_filament_id = self.current_filament_id
        return layer

    def set_layers(self, image_layer: napari.layers.Image):
        self.image_layer = image_layer
        if self.points_layer is None and self.image_layer is not None:
            self.points_layer = self._create_points_layer()
            self.viewer.add_layer(self.points_layer)

    def _on_enable(self):
        add_mouse_callback_safe(
            self.viewer.mouse_drag_callbacks, self._mouse_callback
        )

    def _on_disable(self):
        remove_mouse_callback_safe(
            self.viewer.mouse_drag_callbacks, self._mouse_callback
        )
