from typing import Optional

import napari
from napari.layers.points._points_mouse_bindings import select as napari_selection_callback
import numpy as np

from .base_manipulator import BaseManipulator
from ..utils.napari_utils import remove_mouse_callback_safe, add_mouse_callback_safe


class PointManipulator(BaseManipulator):

    def __init__(self, viewer, layer=None, order=0, translator_length=50, rotator_radius=5):
        self._visible = False
        super().__init__(
            viewer,
            layer,
            order=order,
            translator_length=translator_length,
            rotator_radius=rotator_radius,
            enabled=False
        )

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, visible: bool):
        # if visible == self._visible:
        #     return
        if self.node is not None:
            self.node.visible = visible
        self._visible = visible

    def set_layers(self, layer: napari.layers.Points):
        super().set_layers(layer)

    def _initialize_transform(self):
        self._translation = np.array([0, 0, 0])
        self._rot_mat = np.eye(3)

        if self.layer is not None:
            self._on_selection_change()


    def _set_initial_translation_vectors(self):
        self._initial_translation_vectors_ = np.asarray(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )

    def _set_initial_rotator_normals(self):
        self._initial_rotator_normals_ = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ]
        )

    def _connect_events(self, layer):
        layer.events.highlight.connect(self._on_selection_change)

        if self._layer is not None:
            remove_mouse_callback_safe(
                self._layer.mouse_drag_callbacks,
                napari_selection_callback
            )
            if len(self._layer.selected_data) == 1:
                add_mouse_callback_safe(
                    self._layer.mouse_drag_callbacks,
                    self.napari_selection_callback_passthrough
                )

    def _disconnect_events(self, layer):
        layer.events.highlight.disconnect(self._on_selection_change)
        if self._layer is not None:
            remove_mouse_callback_safe(
                self._layer.mouse_drag_callbacks,
                self.napari_selection_callback_passthrough
            )
            if self._layer.mode == 'select':
                add_mouse_callback_safe(
                    self._layer.mouse_drag_callbacks, napari_selection_callback
                )

    @property
    def active_point_index(self):
        return list(self._layer.selected_data)[0]

    @property
    def active_point_position(self):
        return self._layer.data[self.active_point_index]

    def _on_selection_change(self, event=None):
        if self._layer._is_selecting is True or self.enabled is False:
            return

        selected_points = list(self._layer.selected_data)
        if len(selected_points) == 1:
            remove_mouse_callback_safe(
                self._layer.mouse_drag_callbacks,
                napari_selection_callback
            )
            add_mouse_callback_safe(
                self._layer.mouse_drag_callbacks,
                self.napari_selection_callback_passthrough
            )
            self.visible = True
            self.translation = self.active_point_position

        else:
            self.visible = False
            if self._layer.mode == 'select':
                remove_mouse_callback_safe(
                    self._layer.mouse_drag_callbacks,
                    self.napari_selection_callback_passthrough
                )
                add_mouse_callback_safe(
                    self._layer.mouse_drag_callbacks, napari_selection_callback
                )

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

    def _on_enable(self):
        if self.layer is not None:
            add_mouse_callback_safe(
                self._layer.mouse_drag_callbacks,
                self._mouse_callback,
                index=0
            )
            self._enabled = True
            self._on_selection_change()

    def _on_disable(self):
        if self.layer is not None:
            self.visible = False
            remove_mouse_callback_safe(
                self._layer.mouse_drag_callbacks,
                self._mouse_callback
            )
