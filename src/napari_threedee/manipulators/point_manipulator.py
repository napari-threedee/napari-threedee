import napari
import numpy as np
from napari.layers.points._points_mouse_bindings import select as napari_selection_callback
from napari.layers.points._points_constants import Mode

from .base_manipulator import BaseManipulator
from napari_threedee.utils.napari_utils import remove_mouse_callback_safe, add_mouse_callback_safe


class PointManipulator(BaseManipulator):

    def __init__(self, viewer, layer=None):
        super().__init__(
            viewer,
            layer=layer,
            translator_axes='xyz',
            rotator_axes='xyz'
        )

    def set_layers(self, layer: napari.layers.Points):
        super().set_layers(layer)

    def _initialize_transform(self):
        pass

    def _connect_events(self):
        if self.layer is None:
            return
        self.layer.selected_data.events.items_changed.connect(self._on_selection_change)
        remove_mouse_callback_safe(
            self.layer.mouse_drag_callbacks,
            napari_selection_callback
        )
        if len(self.layer.selected_data) == 1:
            add_mouse_callback_safe(
                self.layer.mouse_drag_callbacks,
                self.napari_selection_callback_passthrough
            )
        self.layer.events.visible.connect(self._on_visibility_change)
        self._viewer.layers.events.removed.connect(self._disable_and_remove)

    def _disconnect_events(self):
        if self.layer is None:
            return
        self.layer.selected_data.events.items_changed.disconnect(self._on_selection_change)
        remove_mouse_callback_safe(
            self.layer.mouse_drag_callbacks,
            self.napari_selection_callback_passthrough
        )
        if self.layer.mode == Mode.SELECT:
            add_mouse_callback_safe(
                self.layer.mouse_drag_callbacks, napari_selection_callback
            )

    @property
    def active_point_index(self):
        return list(self.layer.selected_data)[0]

    @property
    def active_point_position(self) -> np.ndarray:
        """Get the active point position in world coordinates."""
        position_layer_coordinates = self.layer.data[self.active_point_index]
        return np.asarray(self.layer.data_to_world(position_layer_coordinates))

    def _on_selection_change(self, event=None):
        # early exit cases
        if self.layer is None:
            return
        elif self.enabled is False:
            return
        elif self.layer._is_selecting is True:
            return

        selected_points = list(self.layer.selected_data)

        if len(selected_points) == 1:
            # replace napari selection callback with n3d passthrough
            remove_mouse_callback_safe(
                self.layer.mouse_drag_callbacks,
                napari_selection_callback
            )
            add_mouse_callback_safe(
                self.layer.mouse_drag_callbacks,
                self.napari_selection_callback_passthrough
            )
            self.visible = True

            # update manipulator position
            self.origin = self.active_point_position

        else:
            # reinstate original callbacck
            self.visible = False
            if self.layer.mode == Mode.SELECT:
                remove_mouse_callback_safe(
                    self.layer.mouse_drag_callbacks,
                    self.napari_selection_callback_passthrough
                )
                add_mouse_callback_safe(
                    self.layer.mouse_drag_callbacks, napari_selection_callback
                )

    def _pre_drag(self):
        pass

    def _while_dragging_translator(self):
        selected_data = list(self.layer.selected_data)
        if len(selected_data) == 0:
            # return early if no data
            return
        selected_point_index = selected_data[0]
        position_layer_coordinates = self.layer.world_to_data(self.origin)
        self.layer.data[selected_point_index] = position_layer_coordinates
        # # refresh rendering manually after modifying array data inplace
        # with self.layer.events.highlight.blocker():
        self.layer.events.set_data()

        # this is a hack because the transforms were getting overwritten
        # after the set_data() event.
        # https://github.com/napari-threedee/napari-threedee/pull/167
        self._backend._on_transformation_changed()

    def _post_drag(self):
        pass

    # def _while_dragging_rotator(self, selected_rotator: int, rotation_matrix: np.ndarray):
    #     # todo: store rotmat somewhere
    #     pass

    def napari_selection_callback_passthrough(self, layer, event):
        if self._backend.is_dragging:  # early exit if manipulator clicked
            return
        if layer.mode == "pan_zoom":
            # don't change selection if in pan zoom
            return
        # if manipulator not clicked, do normal point selection
        value = layer.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        # updating selected_data triggers an event which reattaches the
        # original napari selection callback.
        if value is not None:
            layer.selected_data = {value}
        else:
            layer.selected_data = set()

    def _on_enable(self):
        super()._on_enable()
        self._on_selection_change()
