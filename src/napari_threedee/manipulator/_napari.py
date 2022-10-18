from typing import Optional, Type, Union

import einops
import napari
import numpy as np
from vispy.visuals.transforms import MatrixTransform

from .axis_model import AxisModel
from .manipulator import ManipulatorModel
from .manipulator_visual_data import ManipulatorVisualData
from .vispy_visual import ManipulatorVisual
from ..manipulators._drag_manager import RotatorDragManager, TranslatorDragManager
from ..utils.napari_utils import get_vispy_node, mouse_event_to_layer_data_displayed, \
    add_mouse_callback_safe
from ..utils.selection_utils import select_sphere_from_click


class NapariManipulator:
    def __init__(
            self,
            translator_axes: str,
            rotator_axes: str,
            viewer: napari.viewer.Viewer,
            layer: Optional[Type[napari.layers.Layer]] = None,
    ):
        self.manipulator = ManipulatorModel.from_strings(
            translators=translator_axes, rotators=rotator_axes
        )
        self.visual_data = ManipulatorVisualData.from_manipulator(self.manipulator)
        self.vispy_visual = ManipulatorVisual(parent=None, manipulator_visual_data=self.visual_data)
        self._viewer = viewer
        self._layer = layer

        if self._layer is not None:
            self._connect_vispy_visual(self._layer)
            self._connect_mouse_callback()
        self.vispy_visual.update()
        self.vispy_visual.update_visuals_from_manipulator_visual_data()

    def _connect_vispy_visual(self, layer: napari.layers.Layer):
        parent = get_vispy_node(self._viewer, layer)
        self.vispy_visual.parent = parent
        self.vispy_visual.transform = MatrixTransform()
        # self.node.order = self._vispy_visual_order

    def _connect_mouse_callback(self):
        add_mouse_callback_safe(
            self._layer.mouse_drag_callbacks,
            self._mouse_callback,
            index=0
        )

    def _mouse_callback(self, layer, event):
        """Mouse call back for selecting and dragging a manipulator."""
        initial_layer_interactive = layer.interactive
        click_position_data_3d, click_dir_data_3d = mouse_event_to_layer_data_displayed(layer, event)
        drag_manager = self._drag_manager_from_click(click_position_data_3d, click_dir_data_3d)
        if drag_manager is None:
            return
        layer.interactive = False  # disable layer interactivity
        yield  # then start handling the mouse drag
        drag_manager.setup_drag(
            layer=layer,
            mouse_event=event,
            translation=self.manipulator.origin,
            rotation_matrix=self.manipulator.rotation_matrix
        )
        while event.type == 'mouse_move':
            new_origin, new_rotation_matrix = drag_manager.update_drag(mouse_event=event)
            print(new_origin, new_rotation_matrix)
            with self.manipulator.events.blocked():
                self.manipulator.origin = new_origin
                self.manipulator.rotation_matrix = new_rotation_matrix
                self._on_transformation_changed()
            yield

        # self.vispy_visual.update_visuals_from_manipulator_visual_data()

        # reset layer interaction to original state
        layer.interactive = initial_layer_interactive

    def _drag_manager_from_click(self, click_point: np.ndarray, view_direction: np.ndarray) -> Optional[Union[RotatorDragManager, TranslatorDragManager]]:
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
        handle_data = self.visual_data.translator_handle_data + self.visual_data.rotator_handle_data
        untransformed_handle_points = einops.rearrange(handle_data.points, 'b xyz -> b xyz 1')
        rotation, translation = self.manipulator.rotation_matrix, einops.rearrange(self.manipulator.origin, 'xyz -> xyz 1')  # Rotation, Translation...
        transformed_handle_points = (rotation @ untransformed_handle_points) + translation
        selection = select_sphere_from_click(
            click_point=click_point,
            view_direction=view_direction,
            sphere_centroids=einops.rearrange(transformed_handle_points, 'b xyz 1 -> b xyz'),
            sphere_diameter=self.visual_data.translator_handle_data.handle_size
        )
        if selection is None:
            return None

        axis_id = handle_data.axis_identifiers[selection]
        self.manipulator.selected_axis_id = axis_id  # triggers highlight change
        axis_vector = AxisModel.from_id(axis_id).vector
        rotated_axis_vector = self.manipulator.rotation_matrix @ axis_vector

        # is the clicked axis a translator or a rotator?
        is_translator = np.zeros(len(handle_data.points), dtype=bool)
        is_translator[:len(self.visual_data.translator_handle_data)] = True
        if is_translator[selection] == True:
            drag_manager = TranslatorDragManager(translation_vector=rotated_axis_vector)
        else:
            drag_manager = RotatorDragManager(rotation_vector=rotated_axis_vector)
        return drag_manager

    def _on_transformation_changed(self) -> None:
        """Update the manipulator visual transformation based on the manipulator state
        """
        if self._layer is None:
            # do not do anything if the layer has not been set
            return
        # convert NumPy axis ordering to VisPy axis ordering
        # by reversing the axes order and flipping the linear
        # matrix
        translation = self.manipulator.origin[::-1]
        rotation_matrix = self.manipulator.rotation_matrix[::-1, ::-1].T

        # Embed in the top left corner of a 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[: rotation_matrix.shape[0], : rotation_matrix.shape[1]] = rotation_matrix
        affine_matrix[-1, : len(translation)] = translation

        self.vispy_visual.transform.matrix = affine_matrix
