import napari
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
        self.layer.events.highlight.connect(self._on_selection_change)
        remove_mouse_callback_safe(
            self.layer.mouse_drag_callbacks,
            napari_selection_callback
        )
        if len(self.layer.selected_data) == 1:
            add_mouse_callback_safe(
                self.layer.mouse_drag_callbacks,
                self.napari_selection_callback_passthrough
            )

    def _disconnect_events(self):
        if self.layer is None:
            return
        self.layer.events.highlight.disconnect(self._on_selection_change)
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
    def active_point_position(self):
        return self.layer.data[self.active_point_index]

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
        self.layer.events.highlight.disconnect(self._on_selection_change)

    def _while_dragging_translator(self):
        selected_point_index = list(self.layer.selected_data)[0]
        self.layer.data[selected_point_index] = self.origin
        # refresh rendering manually after modifying array data inplace
        self.layer.refresh()

    def _post_drag(self):
        self.layer.events.highlight.connect(self._on_selection_change)


    # def _while_dragging_rotator(self, selected_rotator: int, rotation_matrix: np.ndarray):
    #     # todo: store rotmat somewhere
    #     pass

    def napari_selection_callback_passthrough(self, layer, event):
        if self._backend.is_dragging:  # early exit if manipulator clicked
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
