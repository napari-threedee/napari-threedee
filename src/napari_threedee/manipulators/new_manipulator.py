from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type, List, Union

import napari
import numpy as np
from napari.utils.geometry import project_points_onto_plane, rotate_points, rotation_matrix_from_vectors_3d, intersect_line_with_plane_3d
from napari.utils.translations import trans
from napari.viewer import Viewer
from vispy.scene import Line, Compound, Markers
from vispy.visuals.transforms import MatrixTransform

from .manipulator_utils import make_translator_meshes, color_lines, make_rotator_meshes, make_rotator_data
from ..base import ThreeDeeModel
from ..utils.napari_utils import get_vispy_node, add_mouse_callback_safe, remove_mouse_callback_safe, \
    mouse_event_to_layer_data_displayed
from ..utils.selection_utils import select_mesh_from_click, select_sphere_from_click
from ..utils.geometry import signed_angle_between_vectors, rotation_matrix_around_vector_3d

MANIPULATOR_BASIS = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
)

# RGBA colors for each axis
AXIS_COLORS = {
    0: [0.5, 0.8, 0.5, 1],
    1: [0.75, 0.68, 0.83, 1],
    2: [1, 0.75, 0.52, 1]
}

#
CENTRAL_AXIS_VERTICES = {
    0: np.array([[0, 0, 0], [1, 0, 0]]),
    1: np.array([[0, 0, 0], [0, 1, 0]]),
    2: np.array([[0, 0, 0], [0, 0, 1]]),
}

# central axis indices that should be highlighted for a given
# rotator highlight
CENTRAL_AXIS_ROTATOR_HIGHLIGHT_INDICES = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1]
}


class RotatorModel:
    def __init__(
            self,
            normal_vectors: Union[np.ndarray, List[int]],
            colors: np.ndarray,
            radius: float = 20,
            n_segments: int = 64,
            width: float = 5,
            handle_size: float = 3
    ):
        self._normal_vector_indices = normal_vectors
        self.colors = colors
        self.radius = radius
        self.n_segments = n_segments
        self.width = width
        self.handle_size = handle_size

        vertices, connections, colors, handles, rotator_indices = make_rotator_data(
            rotator_normals=self.normal_vectors,
            rotator_colors=self.colors,
            center_point=np.array([0, 0, 0]),
            radius=self.radius,
            n_segments=self.n_segments
        )

        self._vertices = vertices
        self._connections = connections
        self._vertex_colors = colors
        self._handle_points = handles
        self._rotator_indices = rotator_indices

        # array
        self._highlighted_rotators: List[int] = []

    @property
    def normal_vectors(self) -> np.ndarray:
        """Normal vectors for the rotation planes supported by this manipulator.

        Returns
        -------
        normal_vectors : np.ndarray
            The normal vectors in data coordinates for the untransformed manipulator
        """
        return MANIPULATOR_BASIS[self._normal_vector_indices]

    @property
    def vertices(self) -> np.ndarray:
        """Coordinates for the rotator arc vertices.

        Returns
        -------
        vertices : np.ndarray
            (n_rotators * n_segments, 3) array containing the coordinates
            of all vertices.
        """
        return self._vertices

    @property
    def connections(self) -> np.ndarray:
        """Connections between vertices in the rotator arc.

        Returns
        -------
        connections : np.ndarray
            (n_rotators * [n_segments - 1], 2) array containing the
            connections between arc vertices.
        """
        return self._connections

    @property
    def vertex_colors(self) -> np.ndarray:
        """The color for each vertex.

        Returns
        -------
        vertex_colors : np.ndarray
            (n_rotators * n_segments, 4) array containing RGBA colors
            for all vertices.
        """
        return self._vertex_colors


    @property
    def handle_points(self) -> np.ndarray:
        """The coordinates for each handle point.

        Returns
        -------
        handle_points : np.ndarray
            (n_rotators, 3) array containing the coordinates of the handle for
            each rotator.
        """
        return self._handle_points

    @property
    def highlighted_rotators(self) -> List[int]:
        return self._highlighted_rotators

    @highlighted_rotators.setter
    def highlighted_rotators(self, highlighted_rotators: Optional[List[int]]) -> None:
        if highlighted_rotators is None:
            self._highlighted_rotators = []
        else:
            self._highlighted_rotators = list(highlighted_rotators)

    @property
    def rendered_rotator_colors(self) -> Tuple[np.ndarray, np.ndarray]:
        attenuation_factor = 0.5
        if len(self.highlighted_rotators) == 0:
            new_rotator_arc_colors = self.vertex_colors
            new_rotator_handle_colors = self.colors
        else:
            highlight_mask = self.rotator_vertex_mask(self.highlighted_rotators)
            new_rotator_arc_colors = attenuation_factor * self.vertex_colors
            new_rotator_arc_colors[highlight_mask] = self.vertex_colors[highlight_mask]

            new_rotator_handle_colors = attenuation_factor * self.colors
            new_rotator_handle_colors[self.highlighted_rotators] = self.colors[self.highlighted_rotators]

        return new_rotator_arc_colors, new_rotator_handle_colors

    @property
    def n_rotators(self) -> int:
        return len(self.normal_vectors)

    @property
    def rotator_indices(self) -> np.ndarray:
        """The rotator index for each vertex.

        Returns
        -------
        rotator_indices : np.ndarray
            (n_rotators * n_segments,) array containing the rotator index for each vertex.
        """
        return self._rotator_indices

    def rotator_vertex_mask(self, rotator_indices: Union[List[int], int]) -> np.ndarray:
        """Create a boolean mask to select the vertices from the specified rotators.

        Parameters
        ----------
        rotator_indices : Union[List[int], int]
            The indices of the rotators to set to True in the resulting mask.

        Returns
        -------
        vertex_mask : np.ndarray
            The (n_vertices,) array containing True values where the vertices are
            from the selected rotator(s).
        """
        if isinstance(rotator_indices, int):
            rotator_indices = [rotator_indices]

        return np.isin(self.rotator_indices, rotator_indices)


class CentralAxesModel:
    def __init__(self, normal_vectors: Union[np.ndarray, List[int]], colors: np.ndarray, radius: float):
        self._normal_vector_indices: List[int] = normal_vectors
        self.radius = radius
        self.colors = colors

        vertices, connections, colors, axis_indices = self._make_axis_data()
        self._vertices = vertices
        self._connections = connections
        self._vertex_colors = colors
        self._axis_indices = axis_indices

        self._highlighted = []

    def _make_axis_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        vertices = []
        connections = []
        colors = []
        axis_indices = []
        vertex_offset = 0
        for index in self._normal_vector_indices:
            vertices.append(self.radius * CENTRAL_AXIS_VERTICES[index])

            axis_connections = vertex_offset + np.array([0, 1])
            connections.append(axis_connections)

            colors.append(np.tile(AXIS_COLORS[index], (2, 1)))

            # add the rotator indices
            axis_indices.append([index] * 2)

            vertex_offset += 2
        return np.concatenate(vertices, axis=0), np.stack(connections), np.concatenate(colors, axis=0), np.concatenate(axis_indices)

    @property
    def normal_vectors(self) -> np.ndarray:
        """Normal vectors for the central axis .

        Returns
        -------
        normal_vectors : np.ndarray
            The normal vectors in data coordinates for the untransformed manipulator
        """
        return MANIPULATOR_BASIS[self._normal_vector_indices]

    @property
    def vertices(self) -> np.ndarray:
        """Coordinates for the rotator arc vertices.

        Returns
        -------
        vertices : np.ndarray
            (n_rotators * n_segments, 3) array containing the coordinates
            of all vertices.
        """
        return self._vertices

    @property
    def connections(self) -> np.ndarray:
        """Connections between vertices in the rotator arc.

        Returns
        -------
        connections : np.ndarray
            (n_rotators * [n_segments - 1], 2) array containing the
            connections between arc vertices.
        """
        return self._connections

    @property
    def vertex_colors(self) -> np.ndarray:
        """The color for each vertex.

        Returns
        -------
        vertex_colors : np.ndarray
            (n_rotators * n_segments, 4) array containing RGBA colors
            for all vertices.
        """
        return self._vertex_colors

    @property
    def axis_indices(self) -> np.ndarray:
        """The axis index for each vertex.

        Returns
        -------
        axis_indices : np.ndarray
            (n_axes * 2,) array containing the axis index for each vertex.
        """
        return self._axis_indices

    def axis_vertex_mask(self, axis_indices: Union[List[int], int]) -> np.ndarray:
        """Create a boolean mask to select the vertices from the specified rotators.

        Parameters
        ----------
        axis_indices : Union[List[int], int]
            The indices of the rotators to set to True in the resulting mask.

        Returns
        -------
        vertex_mask : np.ndarray
            The (n_vertices,) array containing True values where the vertices are
            from the selected rotator(s).
        """
        if isinstance(axis_indices, int):
            rotator_indices = [axis_indices]

        return np.isin(self.axis_indices, axis_indices)

    @property
    def highlighted(self) -> List[int]:
        return self._highlighted

    @highlighted.setter
    def highlighted(self, highlighted: Optional[List[int]]) -> None:
        if highlighted is None:
            self._highlighted = []
        else:
            self._highlighted = list(highlighted)

    @property
    def rendered_colors(self) -> np.ndarray:
        attenuation_factor = 0.5
        if len(self.highlighted) == 0:
            rendered_axis_colors = self.vertex_colors
        else:
            highlighted_axis_mask = self.axis_vertex_mask(self.highlighted)
            rendered_axis_colors = attenuation_factor * self.vertex_colors
            rendered_axis_colors[highlighted_axis_mask] = self.vertex_colors[highlighted_axis_mask]

        return rendered_axis_colors


class ManipulatorVisual(Compound):
    def __init__(self, parent):
        super().__init__([Line(), Line(), Markers(), Markers()], parent=parent)
        self.centroid_visual.set_data(
            pos=np.array([[0, 0, 0]]),
            face_color=[0.7, 0.7, 0.7, 1],
            size=10
        )
        self.centroid_visual.spherical = True
        self.rotator_handles_visual.spherical = True
        self.rotator_handles_visual.scaling = True
        self.rotator_handles_visual.antialias = 0

    @property
    def axes_visual(self) -> Line:
        return self._subvisuals[0]

    @property
    def rotator_arc_visual(self) -> Line:
        return self._subvisuals[1]

    @property
    def rotator_handles_visual(self) -> Markers:
        return self._subvisuals[2]

    @property
    def centroid_visual(self) -> Markers:
        return self._subvisuals[3]


class RotatorDragManager:

    def __init__(self, normal_vector: np.ndarray, axis_index: int):
        self.rotation_normal = normal_vector
        self.axis_index = axis_index
        self._initial_click_vector = None
        self._initial_rotation_matrix = None
        self._initial_translation = None
        self._view_direction = None

        self._center_on_click_plane = None

    def setup_drag(self, click_point: np.ndarray, view_direction: np.ndarray, translation: np.ndarray, rotation_matrix=np.ndarray):

        click_on_rotation_plane = intersect_line_with_plane_3d(
            line_position=click_point,
            line_direction=view_direction,
            plane_position=translation,
            plane_normal=self.rotation_normal,
        )

        self._initial_click_vector = np.squeeze(click_on_rotation_plane) - translation
        self._initial_rotation_matrix = rotation_matrix.copy()
        self._initial_translation = translation
        self._view_direction = view_direction

    def update_drag(self,
                    click_point,
                    ):
        click_on_rotation_plane = intersect_line_with_plane_3d(
            line_position=click_point,
            line_direction=self._view_direction,
            plane_position=self._initial_translation,
            plane_normal=self.rotation_normal,
        )
        click_vector = np.squeeze(click_on_rotation_plane) - self._initial_translation

        rotation_angle = signed_angle_between_vectors(self._initial_click_vector, click_vector, self.rotation_normal)
        rotation_matrix = rotation_matrix_around_vector_3d(rotation_angle, self.rotation_normal)

        # update the rotation matrix and call the _while_rotator_drag callback
        updated_rotation_matrix = np.dot(rotation_matrix, self._initial_rotation_matrix)

        return self._initial_translation, updated_rotation_matrix


class Manipulator:
    def __init__(
            self,
            viewer: napari.Viewer,
            layer: Optional[Type[napari.layers.Layer]] = None,
            radius: float = 10
    ):
        self._radius = radius

        # initialize the transform
        self._rotation_matrix = np.eye(3)
        self._center_point = np.zeros((3,))

        self._normal_vectors = np.asarray([0, 1, 2])
        colors = np.array([AXIS_COLORS[index] for index in self._normal_vectors])
        self.rotators = RotatorModel(
            normal_vectors=self._normal_vectors,
            colors=colors,
            radius=self._radius
        )

        self.central_axes = CentralAxesModel(normal_vectors=self._normal_vectors, colors=colors, radius=self._radius)

        self.visual = ManipulatorVisual(parent=None)

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

    def _update_central_axes_visual(self):

        self.visual.axes_visual.set_data(
            pos=self.central_axes.vertices[:, ::-1],
            connect=self.central_axes.connections,
            color=self.central_axes.rendered_colors,
            width=10
        )

    def _update_rotators_visual(self):
        rotator_arc_colors, rotator_handle_colors = self.rotators.rendered_rotator_colors
        self.visual.rotator_arc_visual.set_data(
            pos=self.rotators.vertices[:, ::-1],
            connect=self.rotators.connections,
            color=rotator_arc_colors,
            width=self.rotators.width
        )

        # update the handle data
        self.visual.rotator_handles_visual.set_data(
            pos=self.rotators.handle_points[:, ::-1],
            face_color=rotator_handle_colors,
            size=self.rotators.handle_size,
            edge_color=np.array([0, 0, 0, 0])
        )

    def update_visual(self):
        self._update_rotators_visual()
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
        self._initialize_drag(drag_manager, layer, event)

        while event.type == 'mouse_move':
            # process the drag
            click_point_data = np.asarray(layer.world_to_data(event.position))[event.dims_displayed]
            updated_center_point, updated_rotation_matrix = drag_manager.update_drag(
                click_point=click_point_data
            )

            # update the transformation
            # use the private _center_point to avoid the _on_transform_changed()
            # being called twice by the setters
            self._center_point = updated_center_point
            self.rotation_matrix = updated_rotation_matrix
            yield

        # unhighlight the rotator
        self.rotators.highlighted_rotators = None
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
        selected_translator, selected_rotator = self._check_if_manipulator_clicked(
            plane_point=click_position,
            plane_normal=view_direction,
        )

        if selected_rotator is not None:
            self.rotators.highlighted_rotators = [selected_rotator]

            self.central_axes.highlighted = CENTRAL_AXIS_ROTATOR_HIGHLIGHT_INDICES[selected_rotator]
        else:
            self.rotators.highlighted_rotators = None
            self.central_axes.highlighted = None
        self.update_visual()

        if selected_rotator is None:
            return None

        untransformed_normal_vector = self.rotators.normal_vectors[selected_rotator]
        normal_vector = self.rotation_matrix.dot(untransformed_normal_vector)

        drag_manager = RotatorDragManager(normal_vector=normal_vector, axis_index=selected_rotator)

        return drag_manager

    def _initialize_drag(self, drag_manager: RotatorDragManager, layer: napari.layers.Layer, mouse_event) -> None:
        """Prepare the drag manager to start the drag.

        This step sets the initial click position. The drag is relative to this initial position.
        """
        click_point_data, click_dir_data_3d = mouse_event_to_layer_data_displayed(layer, mouse_event)
        drag_manager.setup_drag(
            click_point=click_point_data,
            view_direction=click_dir_data_3d,
            translation=self.center_point,
            rotation_matrix=self.rotation_matrix
        )

    def _check_if_manipulator_clicked(
            self,
            plane_point: np.ndarray,
            plane_normal: np.ndarray
    ) -> Tuple[Optional[int], Optional[int]]:
        """Determine if a translator or rotator was clicked on.

        Parameters
        ----------
        plane_point : np.ndarray
            The click point in data coordinates
        plane_normal : np.ndarray
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
        # project the in view points onto the plane
        # if len(self.translator_normals) > 0:
        #     translator_triangles = self._displayed_translator_vertices[self.translator_indices]
        #     selected_translator = select_mesh_from_click(
        #         click_point=plane_point,
        #         view_direction=plane_normal,
        #         triangles=translator_triangles,
        #         triangle_indices=self.translator_triangle_indices
        #     )
        # else:
        #     selected_translator = None
        selected_translator = None

        if self.rotators.n_rotators > 0:
            selected_rotator = self._check_rotator_handle_clicked(click_point=plane_point, view_direction=plane_normal)
        else:
            selected_rotator = None

        return selected_translator, selected_rotator

    def _check_rotator_handle_clicked(self, click_point: np.ndarray, view_direction: np.ndarray) -> Optional[int]:
        """Determine which, if any, rotator was clicked.

        Parameters
        ----------
        click_point : np.ndarray
            The point where the click was performed in displayed data coordinates.
        view_direction : np.ndarray
            A unit vector in the direction of the view ray in displayed data coordinates.

        Returns
        -------
        selection : Optional[int]
            The index of the rotator that was selected.
            Returns None if no rotator handle was clicked.
        """
        # get the handle points in the current manipulator configuration
        untransformed_handle_points = self.rotators.handle_points
        current_handle_points = untransformed_handle_points @ self.rotation_matrix.T

        selection = select_sphere_from_click(
            click_point=click_point,
            view_direction=view_direction,
            sphere_centroids=current_handle_points,
            sphere_diameter=self.rotators.handle_size
        )

        return selection

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
        affine_matrix[: rotation_matrix.shape[0], : rotation_matrix.shape[1]] = rotation_matrix
        affine_matrix[-1, : len(translation)] = translation

        self.visual.transform.matrix = affine_matrix
