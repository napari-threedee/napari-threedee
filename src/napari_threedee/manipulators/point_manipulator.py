from typing import Optional

from napari.layers.points._points_mouse_bindings import select as napari_selection_callback
import numpy as np

from .base_manipulator import BaseManipulator
from ..utils.napari_utils import remove_mouse_callback_safe, add_mouse_callback_safe


class PointManipulator(BaseManipulator):

    def __init__(self, viewer, layer, order=0, translator_length=50, rotator_radius=5):
        self._layer = layer
        self._connect_events(layer)
        self._translation = [0, 0, 0]
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
            rotator_radius=rotator_radius,
            visible=False
        )

        self._on_selection_change()

    def _connect_events(self, layer):
        layer.events.highlight.connect(self._on_selection_change)

    def _remove_events(self, layer):
        layer.events.highlight.disconnect(self._on_selection_change)

    @property
    def active_point_index(self):
        return list(self._layer.selected_data)[0]

    @property
    def active_point_position(self):
        return self._layer.data[self.active_point_index]

    def _on_selection_change(self, event=None):
        if self._layer._is_selecting is True:
            return

        selected_points = list(self._layer.selected_data)
        if len(selected_points) == 1:
            if napari_selection_callback in self._layer.mouse_drag_callbacks:
                remove_mouse_callback_safe(self._layer, napari_selection_callback)
                add_mouse_callback_safe(self._layer, self.napari_selection_callback_passthrough)
            self.visible = True
            self.translation = self.active_point_position

        else:
            self.visible = False
            if self._layer.mode == 'select':
                remove_mouse_callback_safe(self._layer, self.napari_selection_callback_passthrough)
                add_mouse_callback_safe(self._layer, napari_selection_callback)



    def _pre_drag(
            self,
            click_point: np.ndarray,
            selected_translator: Optional[int],
            selected_rotator: Optional[int]
    ):
        pass

    def _while_dragging_translator(self, selected_translator: int, translation_vector: np.ndarray):
        self._layer._move([self.active_point_index], self.translation)
        self._drag_start = None

    def _while_dragging_rotator(self, selected_rotator: int, rotation_matrix: np.ndarray):
        # todo: store rotmat somewhere
        pass

    def _post_drag(self):
        pass

    def napari_selection_callback_passthrough(self, layer, event):
        # get click position and direction in data coordinates
        click_position_world = event.position

        click_position_data_3d = np.asarray(
            self._layer._world_to_displayed_data(
                click_position_world,
                event.dims_displayed
            )
        )
        click_dir_data_3d = np.asarray(
            self._layer._world_to_displayed_data_ray(
                event.view_direction,
                event.dims_displayed
            )
        )

        # identify clicked rotator/translator
        selected_translator, selected_rotator = self._check_if_manipulator_clicked(
            plane_point=click_position_data_3d,
            plane_normal=click_dir_data_3d,
        )
        manipulator_clicked = (selected_translator is not None) or (selected_rotator is not None)

        if manipulator_clicked:  # early exit
            return

        # reestablish normal behaviour
        # layer.selected_data changing emits an event which reattaches the
        # original napari_selection_callback
        value = layer.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        if value is not None:
            layer.selected_data = {value}
        else:
            layer.selected_data = set()
