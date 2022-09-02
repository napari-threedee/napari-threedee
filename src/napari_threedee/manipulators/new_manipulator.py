from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type, List, Union

import napari
import numpy as np
from napari.utils.geometry import project_points_onto_plane, rotate_points
from napari.utils.translations import trans
from napari.viewer import Viewer
from vispy.scene import Line, Compound, Markers
from vispy.visuals.transforms import MatrixTransform

from .manipulator_utils import make_translator_meshes, color_lines, make_rotator_meshes, make_rotator_data
from ..base import ThreeDeeModel
from ..utils.napari_utils import get_vispy_node, add_mouse_callback_safe, remove_mouse_callback_safe
from ..utils.selection_utils import select_mesh_from_click, select_sphere_from_click


class RotatorModel:
    def __init__(
            self,
            normal_vectors: np.ndarray,
            colors: np.ndarray,
            radius: float = 20,
            n_segments: int = 64,
            width: float = 5,
            handle_size: float = 3
    ):
        self.normal_vectors = normal_vectors
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


class Manipulator:
    def __init__(
            self,
            viewer: napari.Viewer,
            layer: Optional[Type[napari.layers.Layer]] = None,
            radius: float = 10
    ):
        self._radius = radius

        # initialize the transform
        self.rotation_matrix = np.eye(3)
        self.center_point = np.zeros((3,))

        normal_vectors = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]
        )
        colors = np.array(
            [
                [0.5, 0.8, 0.5, 1],
                [0.75, 0.68, 0.83, 1],
                [1, 0.75, 0.52, 1]
            ]
        )
        self.rotators = RotatorModel(
            normal_vectors=normal_vectors,
            colors=colors,
            radius=self._radius
        )

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

    def _connect_vispy_visual(self, layer: Type[napari.layers.Layer]):
        # get the callback_list node to pass as the parent of the manipulator
        parent = get_vispy_node(self._viewer, layer)
        self.visual.parent = parent

        self.visual.transform = MatrixTransform()
        # self.node.order = self._vispy_visual_order

    def _update_central_axes_visual(self):
        vertices = self._radius * np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 1],
            ]
        )
        connections = np.array(
            [
                [0, 1],
                [2, 3],
                [4, 5]
            ]
        )
        colors = np.array(
            [
                [0.5, 0.8, 0.5, 1],
                [0.5, 0.8, 0.5, 1],
                [0.75, 0.68, 0.83, 1],
                [0.75, 0.68, 0.83, 1],
                [1, 0.75, 0.52, 1],
                [1, 0.75, 0.52, 1]
            ]
        )
        self.visual.axes_visual.set_data(
            pos=vertices,
            connect=connections,
            color=colors,
            width=10
        )

    def _update_rotators_visual(self):
        self.visual.rotator_arc_visual.set_data(
            pos=self.rotators.vertices,
            connect=self.rotators.connections,
            color=self.rotators.vertex_colors,
            width=self.rotators.width
        )

        # update the handle data
        self.visual.rotator_handles_visual.set_data(
            pos=self.rotators.handle_points,
            face_color=self.rotators.colors,
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

        print(selected_rotator)
        self.highlight_rotator(selected_rotator)

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

    def highlight_rotator(self, rotator_indices: Optional[Union[List[int], int]] = None) -> None:
        attenuation_factor = 0.5
        if rotator_indices is None:
            new_rotator_arc_colors = self.rotators.vertex_colors
            new_rotator_handle_colors = self.rotators.colors
        else:
            highlight_mask = self.rotators.rotator_vertex_mask(rotator_indices)
            new_rotator_arc_colors = attenuation_factor * self.rotators.vertex_colors
            new_rotator_arc_colors[highlight_mask] = self.rotators.vertex_colors[highlight_mask]

            new_rotator_handle_colors = attenuation_factor * self.rotators.colors
            new_rotator_handle_colors[rotator_indices] = self.rotators.colors[rotator_indices]

        # set the new colors
        # self.visual.rotator_arc_visual.set_data(
        #     pos=self.rotators.vertices,
        #     connect=self.rotators.connections,
        #     color=new_rotator_arc_colors,
        #     width=self.rotators.width
        # )
        self.visual.rotator_arc_visual.set_data(
            color=new_rotator_arc_colors,
        )
        self.visual.rotator_handles_visual.set_data(
            pos=self.rotators.handle_points,
            face_color=new_rotator_handle_colors,
            size=self.rotators.handle_size,
            edge_color=np.array([0, 0, 0, 0])
        )
