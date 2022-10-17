from typing import Optional, Type, Union

import napari
import numpy as np
from pydantic import BaseModel
from vispy.visuals.transforms import MatrixTransform

from napari_threedee.manipulator.manipulator import ManipulatorModel
from napari_threedee.manipulators._drag_manager import RotatorDragManager, TranslatorDragManager
from napari_threedee.manipulators._manipulator_visual import ManipulatorVisual
from napari_threedee.utils.napari_utils import get_vispy_node, add_mouse_callback_safe, mouse_event_to_layer_data_displayed
from napari_threedee.utils.selection_utils import select_sphere_from_click


class Manipulator(BaseModel):
    model: ManipulatorModel
    viewer: napari.viewer.Viewer
    visual: ManipulatorVisual = ManipulatorVisual(parent=None)
    layer: Optional[napari.layers.Layer] = None



        self._viewer = viewer
        self._layer = layer
        if self._layer is not None:
            self._connect_vispy_visual(self._layer)

        self.update_visual()

        add_mouse_callback_safe(
            self._layer.mouse_drag_callbacks,
            self._mouse_callback,
            index=0
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self._rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix: np.ndarray) -> None:
        self._rotation_matrix = rotation_matrix
        self._on_transformation_changed()

    @property
    def center_point(self) -> np.ndarray:
        return self._center_point

    @center_point.setter
    def center_point(self, center_point) -> None:
        self._center_point = center_point
        self._on_transformation_changed()

    def _connect_vispy_visual(self, layer: Type[napari.layers.Layer]) -> None:
        # get the callback_list node to pass as the parent of the manipulator
        parent = get_vispy_node(self._viewer, layer)
        self.visual.parent = parent

        self.visual.transform = MatrixTransform()
        # self.node.order = self._vispy_visual_order

    def _update_central_axes_visual(self) -> None:
        # set the axes
        self.visual.axes_visual.set_data(
            pos=self.central_axes.vertices[:, ::-1],
            connect=self.central_axes.connections,
            color=self.central_axes.rendered_vertex_colors,
            width=10
        )

        # set the accenting points that cap the axes
        self.visual.centroid_visual.set_data(
            pos=np.array([[0, 0, 0]]),
            face_color=[0.7, 0.7, 0.7, 1],
            size=10
        )

    def _update_rotators_visual(self) -> None:
        rotator_arc_colors, rotator_handle_colors = self.rotators.rendered_rotator_colors
        self.visual.rotator_arc_visual.set_data(
            pos=self.rotators.vertices[:, ::-1],
            connect=self.rotators.connections,
            color=rotator_arc_colors,
            width=self.rotators.line_width
        )

        # update the handle data
        self.visual.rotator_handles_visual.set_data(
            pos=self.rotators.handle_points[:, ::-1],
            face_color=rotator_handle_colors,
            size=self.rotators.handle_size,
            edge_color=np.array([0, 0, 0, 0])
        )

    def _update_translators_visual(self) -> None:
        translator_axis_colors, translator_handle_colors = self.translators.rendered_translator_colors

        # set the axes
        self.visual.translator_visual.set_data(
            pos=self.translators.vertices[:, ::-1],
            color=translator_axis_colors,
            connect="segments",
            width=self.translators.line_width,
        )

        # set the handles
        self.visual.translator_handles_visual.set_data(
            pos=self.translators.handle_points[:, ::-1],
            face_color=translator_handle_colors, size=self.translators.handle_size, edge_color=np.array([0, 0, 0, 0])
        )

    def update_visual(self):
        self._update_rotators_visual()
        self._update_translators_visual()
        self._update_central_axes_visual()

    def _mouse_callback(self, layer, event):
        """Mouse call back for selecting and dragging a manipulator."""
        # get the initial state for layer.interactive so we can return
        # at the end of the callback
        initial_layer_interactive = layer.interactive

        # get click position and direction in data coordinates
        click_position_data_3d, click_dir_data_3d = mouse_event_to_layer_data_displayed(layer, event)

        # determine which, if any rotator/translator was selected
        drag_manager = self._process_mouse_event_click(click_position_data_3d, click_dir_data_3d)

        if drag_manager is None:
            # return early if no layer was selected
            return

        # set turn off interaction so drags don't move the camera
        layer.interactive = False

        yield

        # start the drag
        drag_manager.setup_drag(
            layer=layer,
            mouse_event=event,
            translation=self.center_point,
            rotation_matrix=self.rotation_matrix
        )

        while event.type == 'mouse_move':
            # process the drag
            updated_center_point, updated_rotation_matrix = drag_manager.update_drag(
                mouse_event=event
            )

            # update the transformation
            # use the private _center_point to avoid the _on_transform_changed()
            # being called twice by the setters
            self._center_point = updated_center_point
            self.rotation_matrix = updated_rotation_matrix
            yield

        # unhighlight the manipulator components
        self.rotators.highlighted_rotators = None
        self.translators.highlighted_translators = None
        self.central_axes.highlighted = None
        self.update_visual()

        # reset layer interaction to original state
        layer.interactive = initial_layer_interactive

    def _process_mouse_event_click(self, click_position: np.ndarray, view_direction: np.ndarray) -> Optional[RotatorDragManager]:
        """Determine which rotator or translator was selected and return the appropriate drag manager.

        Parameters
        ----------
        click_position : np.ndarray
            The click position in displayed data coordinates.
        view_direction : np.ndarray
            The vector pointing in the direction of the camera in displayed data coordinates.

        Returns
        -------
        drag_manager : Optional[RotatorDragManager]
            The DragManager object for the translator or rotator selected.
            Returns None if no rotator or translator was clicked.
        """

        # identify clicked rotator/translator
        drag_manager = self._check_if_manipulator_clicked(click_point=click_position, view_direction=view_direction)

        if drag_manager is not None:
            if isinstance(drag_manager, RotatorDragManager):
                self.rotators.highlighted_rotators = [drag_manager.axis_index]
                self.translators.highlighted_translators = []
                self.central_axes.highlighted = PERPENDICULAR_AXES[drag_manager.axis_index]
            else:
                self.rotators.highlighted_rotators = []
                self.translators.highlighted_translators = [drag_manager.axis_index]
                self.central_axes.highlighted = [drag_manager.axis_index]
        else:
            self.rotators.highlighted_rotators = None
            self.central_axes.highlighted = None
        self.update_visual()

        return drag_manager

    def _check_if_manipulator_clicked(self, click_point: np.ndarray, view_direction: np.ndarray) -> Optional[Union[RotatorDragManager, TranslatorDragManager]]:
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
        # concatentate and transform all handle points
        untransformed_handle_points = np.concatenate(
            [self.rotators.handle_points,
             self.translators.handle_points]
        )
        current_handle_points = (untransformed_handle_points @ self.rotation_matrix.T) + self.center_point
        selection = select_sphere_from_click(
            click_point=click_point,
            view_direction=view_direction,
            sphere_centroids=current_handle_points,
            sphere_diameter=self.rotators.handle_size
        )

        if selection is None:
            return None

        n_rotators = self.rotators.n_rotators
        if selection < n_rotators:
            # a rotator was selected
            untransformed_normal_vector = self.rotators.basis_vectors[selection]
            normal_vector = self.rotation_matrix.dot(untransformed_normal_vector)
            return RotatorDragManager(normal_vector=normal_vector, axis_index=selection)
        else:
            # a translator was selected
            translator_index = selection - n_rotators
            untransformed_normal_vector = self.translators.basis_vectors[translator_index]
            normal_vector = self.rotation_matrix.dot(untransformed_normal_vector)
            return TranslatorDragManager(normal_vector=normal_vector, axis_index=translator_index)

    def _on_transformation_changed(self) -> None:
        """Update the manipulator visual transformation based on the
        manipulator state
        """
        if self._layer is None:
            # do not do anything if the layer has not been set
            return
        # convert NumPy axis ordering to VisPy axis ordering
        # by reversing the axes order and flipping the linear
        # matrix
        translation = self.center_point[::-1]
        rotation_matrix = self.rotation_matrix[::-1, ::-1].T

        # Embed in the top left corner of a 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = rotation_matrix
        affine_matrix[-1, :3] = translation

        self.visual.transform.matrix = affine_matrix
