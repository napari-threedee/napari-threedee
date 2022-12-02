from typing import Optional, Type, Union

import einops
import napari
import numpy as np
from vispy.visuals.transforms import MatrixTransform

from .axis_model import AxisModel
from .manipulator_model import ManipulatorModel
from .vispy_visual_data import ManipulatorVisualData
from .vispy_manipulator_visual import ManipulatorVisual
from napari_threedee._backend.manipulator.drag_managers import RotatorDragManager, \
    TranslatorDragManager
from ...utils.napari_utils import get_vispy_node, \
    get_mouse_position_in_displayed_layer_data_coordinates, \
    add_mouse_callback_safe, remove_mouse_callback_safe
from ...utils.selection_utils import select_sphere_from_click


class NapariManipulatorBackend:
    def __init__(
            self,
            translator_axes: str,
            rotator_axes: str,
            viewer: napari.viewer.Viewer,
            layer: Optional[Type[napari.layers.Layer]] = None,
    ):
        self.manipulator_model = ManipulatorModel.from_strings(
            translators=translator_axes, rotators=rotator_axes
        )
        self.vispy_visual_data = ManipulatorVisualData.from_manipulator(self.manipulator_model)
        self.vispy_visual = ManipulatorVisual(parent=None,
                                              manipulator_visual_data=self.vispy_visual_data)
        self._viewer = viewer
        self._layer = layer
        self.is_dragging = False

        if self._layer is not None:
            self._connect_vispy_visual()
            self._connect_mouse_callback()

        self._connect_transformation_events()
        self.vispy_visual.update()
        self.vispy_visual.update_visuals_from_manipulator_visual_data()

    @property
    def layer(self) -> napari.layers.Layer:
        return self._layer

    @layer.setter
    def layer(self, layer: napari.layers.Layer):
        self.vispy_visual.parent = None
        if self._layer is not None:
            self._disconnect_mouse_callback()
        self._layer = layer
        self._connect_vispy_visual()
        self._connect_mouse_callback()

    @property
    def is_dragging(self) -> bool:
        return self._is_dragging

    @is_dragging.setter
    def is_dragging(self, value: bool):
        self._is_dragging = value

    def _connect_vispy_visual(self):
        parent = get_vispy_node(self._viewer, self.layer)
        self.vispy_visual.parent = parent
        self.vispy_visual.transform = MatrixTransform()
        self.vispy_visual.canvas._backend.destroyed.connect(self._set_canvas_none)

    def _connect_transformation_events(self):
        # updating the model should update the view
        self.manipulator_model.events.origin.connect(self._on_transformation_changed)

    def _connect_mouse_callback(self):
        add_mouse_callback_safe(
            self._layer.mouse_drag_callbacks,
            self._mouse_callback,
            index=0
        )

    def _disconnect_mouse_callback(self):
        remove_mouse_callback_safe(
            self.layer.mouse_drag_callbacks,
            self._mouse_callback
        )

    def _update_colors(self):
        if self.manipulator_model.selected_axis_id is None:
            self.vispy_visual_data.selected_axes = []
        else:
            self.vispy_visual_data.selected_axes = [self.manipulator_model.selected_axis_id]
        self.vispy_visual.update_visuals_from_manipulator_visual_data()

    def _mouse_callback(self, layer, event):
        """Mouse call back for selecting and dragging a manipulator."""
        initial_layer_interactive = layer.interactive
        click_position_data_3d, click_dir_data_3d = get_mouse_position_in_displayed_layer_data_coordinates(
            layer, event
        )
        drag_manager = self._drag_manager_from_click(click_position_data_3d, click_dir_data_3d)
        if drag_manager is None:  # no translator/rotator was clicked
            return

        # setup...
        layer.interactive = False
        self.is_dragging = True
        self._update_colors()

        yield  # then start handling the mouse drag
        drag_manager.setup_drag(
            layer=layer,
            mouse_event=event,
            translation=self.manipulator_model.origin,
            rotation_matrix=self.manipulator_model.rotation_matrix
        )
        while event.type == 'mouse_move':
            new_origin, new_rotation_matrix = drag_manager.update_drag(mouse_event=event)
            with self.manipulator_model.events.blocked():
                self.manipulator_model.origin = new_origin
                self.manipulator_model.rotation_matrix = new_rotation_matrix
                self._on_transformation_changed()
            yield

        # reset manipulator visual and layer interactivity to original state
        self.manipulator_model.selected_axis_id = None
        self._update_colors()
        layer.interactive = initial_layer_interactive
        self.is_dragging = False

    def _drag_manager_from_click(
            self, click_point: np.ndarray, view_direction: np.ndarray
    ) -> Optional[Union[RotatorDragManager, TranslatorDragManager]]:
        """Determine if a translator or rotator was clicked on.
        Parameters
        ----------
        click_point : np.ndarray
            The click point in data coordinates
        view_direction : np.ndarray
            The vector in the direction of the view (click).
        Returns
        -------
        selected_translator : Optional[int]
            If a translator was clicked, returns the index of the translator.
            If no translator was clicked, returns None.
        selected_rotator : Optional[int]
            If a rotator was clicked, returns the index of the rotator.
            If no rotator was clicked, returns None.
        """
        handle_data = self.vispy_visual_data.translator_handle_data + self.vispy_visual_data.rotator_handle_data
        untransformed_handle_points = einops.rearrange(handle_data.points, 'b xyz -> b xyz 1')
        rotation_matrix = self.manipulator_model.rotation_matrix
        translation = einops.rearrange(self.manipulator_model.origin, 'xyz -> xyz 1')
        transformed_handle_points = (rotation_matrix @ untransformed_handle_points) + translation
        selection = select_sphere_from_click(
            click_point=click_point,
            view_direction=view_direction,
            sphere_centroids=einops.rearrange(transformed_handle_points, 'b xyz 1 -> b xyz'),
            sphere_diameter=self.vispy_visual_data.translator_handle_data.handle_size
        )
        if selection is None:
            return None

        selected_axis_id = handle_data.axis_identifiers[selection]
        self.manipulator_model.selected_axis_id = selected_axis_id
        axis_vector = AxisModel.from_id(selected_axis_id).vector
        rotated_axis_vector = self.manipulator_model.rotation_matrix @ axis_vector

        # is the clicked point  a translator or a rotator?
        point_is_translator = np.zeros(len(handle_data.points), dtype=bool)
        point_is_translator[:len(self.vispy_visual_data.translator_handle_data)] = True
        point_is_translator = point_is_translator[selection] == True  # np.array(True) is not True
        if point_is_translator:
            self.manipulator_model.selected_object_type = 'translator'
            self.vispy_visual_data.translator_is_selected = True
            drag_manager = TranslatorDragManager(translation_vector=rotated_axis_vector)
        else:
            self.manipulator_model.selected_object_type = 'rotator'
            self.vispy_visual_data.rotator_is_selected = True
            drag_manager = RotatorDragManager(rotation_vector=rotated_axis_vector)
        return drag_manager

    def _on_transformation_changed(self) -> None:
        """Update the manipulator visual transformation based on the manipulator state
        """
        if self._layer is None:
            return
        # convert NumPy axis ordering to VisPy axis ordering
        translation = self.manipulator_model.origin[::-1]
        rotation_matrix = self.manipulator_model.rotation_matrix[::-1, ::-1].T

        # Embed rotation matrix in the top left corner of a 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[: rotation_matrix.shape[0], : rotation_matrix.shape[1]] = rotation_matrix
        affine_matrix[-1, : len(translation)] = translation

        # update transform on vispy manipulator
        self.vispy_visual.transform.matrix = affine_matrix

    def _on_ndisplay_change(self, event=None):
        if self._viewer.dims.ndisplay == 2:
            self._disconnect_mouse_callback()
        else:
            self._connect_mouse_callback()

    def _set_canvas_none(self):
        self.vispy_visual._set_canvas(None)
