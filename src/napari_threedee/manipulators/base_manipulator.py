from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from napari.utils.geometry import project_points_onto_plane, rotation_matrix_from_vectors
from napari.utils.translations import trans
from vispy.scene import Mesh
from vispy.visuals.transforms import MatrixTransform


from .manipulator_utils import make_translator_meshes, select_rotator, color_lines, make_rotator_meshes
from ..utils.selection_utils import select_line_segment, select_mesh_from_click
from ..utils.napari_utils import get_vispy_node, get_napari_visual


class BaseManipulator(ABC):
    """Base class for manipulators.

    To implement:
        1. Define self._initial_translator_normals in the __init__. This a
        (Nx3) numpy array containing the normal direction for each of the N
        translators to be created defined in displayed data coordinates.
        2. Define self._initial_rotator_normals in the __init__. This a
        (Nx3) numpy array containing the normal direction for each of the N
        rotators to be created defined in displayed data coordinates.
        3. Call the super.__init__() last.
        4. Implement the drag callback functions
    """
    _N_SEGMENTS_ROTATOR = 50
    _N_TUBE_POINTS = 15
    def __init__(
            self,
            viewer,
            layer=None,
            order=0,
            translator_length=50,
            translator_width=1,
            rotator_radius=5,
            rotator_width=1,
    ):
        super().__init__()
        self._viewer = viewer
        self._layer = layer

        self._translator_length = translator_length
        self._translator_width = translator_width
        self._rotator_radius = rotator_radius
        self._rotator_width = rotator_width

        self._layer.mouse_drag_callbacks.append(self.on_click)

        # this is used to store the vector to the initial click
        # on a rotator for calculating the rotation
        self._initial_click_vector = None

        # Initialize the rotation matrix describing the orientation of the manipulator.
        self._rot_mat = np.eye(3)

        # CMYRGB for 6 axes data in x, y, z, ... ordering
        self._default_color = [
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]

        # initialize the arrow lines. if they were defined by the super class,
        # initialize them as empty.
        if not hasattr(self, '_initial_translator_normals'):
            self._initial_translator_normals = np.empty((0, 3))
        self._init_translators()

        # initialize the rotators. if they were defined by the super class,
        # initialize them as empty.
        if not hasattr(self, '_initial_rotator_normals'):
            self._initial_rotator_normals = np.empty((0, 3))
        self._init_rotators()

        # get the layer node to pass as the parent of the manipulator
        parent = get_vispy_node(viewer, layer)
        self.node = Mesh(mode='triangles', shading='smooth', parent=parent)

        self.node.transform = MatrixTransform()
        self.node.order = order

        self.node.canvas._backend.destroyed.connect(self._set_canvas_none)

        self._viewer.camera.events.zoom.connect(self._on_zoom_change)

        self._on_matrix_change()
        self._on_data_change()
        self.node.visible = True

    def _init_translators(self):
        translator_vertices, translator_indices, translator_colors, triangle_indices = make_translator_meshes(
            centroids=np.asarray([0, 0, 0]),
            normals=self._initial_translator_normals,
            colors=self._default_color[:len(self._initial_translator_normals)],
            translator_length=self.translator_length,
            tube_radius=self._translator_width,
            tube_points=self._N_TUBE_POINTS,
        )

        self.translator_vertices = translator_vertices
        self.translator_indices = translator_indices
        self.translator_colors = translator_colors
        self.translator_triangle_indices = triangle_indices

        self._translator_normals = self._initial_translator_normals.copy()

    def _init_rotators(self):
        if len(self._initial_rotator_normals) == 0:
            return None
        rotator_vertices, rotator_indices, rotator_colors, triangle_indices = make_rotator_meshes(
            centroids=np.repeat([0, 0, 0], 3, axis=0),
            normals=self._initial_rotator_normals,
            colors=self._default_color[:len(self._initial_rotator_normals)],
            rotator_radius=self.rotator_radius,
            tube_radius=1,
            tube_points=self._N_TUBE_POINTS,
            n_segments=self._N_SEGMENTS_ROTATOR
        )

        self.rotator_vertices = rotator_vertices
        self.rotator_indices = rotator_indices
        self.rotator_colors = rotator_colors
        self.rotator_triangle_indices = triangle_indices

        self._rotator_normals = self._initial_rotator_normals.copy()

    @property
    def centroid(self) -> np.ndarray:
        return self._centroid

    @centroid.setter
    def centroid(self, centroid: np.ndarray):
        self._centroid = centroid
        self._on_matrix_change()

    @property
    def rot_mat(self) -> np.ndarray:
        return self._rot_mat

    @rot_mat.setter
    def rot_mat(self, rotation_matrix: np.ndarray):
        self._rot_mat = rotation_matrix
        self._on_matrix_change()

    @property
    def translator_length(self) -> float:
        return self._translator_length

    @translator_length.setter
    def translator_length(self, line_length: float):
        self._translator_length = line_length
        self._on_data_change()

    @property
    def translator_width(self) -> float:
        return self._translator_width

    @translator_width.setter
    def translator_width(self, width: float):
        self._translator_width = width
        self._update_translator_mesh()

    @property
    def rotator_radius(self) -> float:
        return self._rotator_radius

    @rotator_radius.setter
    def rotator_radius(self, rotator_radius):
        self._rotator_radius = rotator_radius
        self._on_data_change()

    @property
    def rotator_width(self) -> float:
        return self._rotator_width

    @rotator_width.setter
    def rotator_width(self, width: float):
        self._rotator_width = width
        self.update_rotator_mesh()

    @property
    def translator_normals(self) -> np.ndarray:
        return (self._initial_translator_normals @ self.rot_mat.T)

    @property
    def rotator_normals(self) -> np.ndarray:
        return (self._initial_rotator_normals @ self.rot_mat.T)

    @property
    def displayed_translator_vertices(self):
        if self.translator_vertices is not None:
            return (self.translator_vertices @ self.rot_mat.T) + self.centroid
        else:
            return None

    @property
    def displayed_rotator_vertices(self):
        if self.rotator_vertices is not None:
            return (self.rotator_vertices @ self.rot_mat.T) + self.centroid
        else:
            return None

    def on_click(self, layer, event):
        """Mouse call back for selecting and dragging an axis"""

        # get the points and vectors in data coordinates
        point_world = event.position
        point_data = np.asarray(self._layer.world_to_data(point_world))
        plane_point = point_data[event.dims_displayed]

        view_dir_data = np.asarray(self._layer._world_to_data_ray(event.view_direction))
        plane_normal = view_dir_data[event.dims_displayed]

        # project the in view points onto the plane
        if len(self.translator_normals) > 0:
            translator_triangles = self.displayed_translator_vertices[self.translator_indices]
            selected_translator = select_mesh_from_click(
                click_point=plane_point,
                view_direction=plane_normal,
                triangles=translator_triangles,
                triangle_indices=self.translator_triangle_indices
            )
            if selected_translator is not None:
                selected_translator_normal = self.translator_normals[selected_translator]
                layer.interactive = False
            else:
                selected_translator_normal = None
        else:
            selected_translator = None

        if len(self.rotator_normals) > 0:
            rotator_triangles = self.displayed_rotator_vertices[self.rotator_indices]
            selected_rotator = select_mesh_from_click(
                click_point=plane_point,
                view_direction=plane_normal,
                triangles=rotator_triangles,
                triangle_indices=self.rotator_triangle_indices
            )
            if selected_rotator is not None:
                layer.interactive = False
        else:
            selected_rotator = None

        initial_position_world = event.position
        yield

        if selected_translator is not None or selected_rotator is not None:

            self._setup_translator_drag(
                click_point=plane_point, selected_translator=selected_translator
            )
            # set up for the mouse drag
            self._setup_rotator_drag(
                click_point=plane_point, selected_rotator=selected_rotator
            )

            # call the _pre_drag callback
            self._pre_drag(
                click_point=plane_point,
                selected_translator=selected_translator,
                selected_rotator=selected_rotator
            )

            while event.type == 'mouse_move':
                # click position
                coordinates = np.asarray(layer.world_to_data(event.position))[event.dims_displayed]

                rotator_drag_vector = None
                translator_drag_vector = None
                if selected_translator is not None:
                    # get drag vector projected onto the translator axis
                    projected_distance = layer.projected_distance_from_mouse_drag(
                        start_position=initial_position_world,
                        end_position=event.position,
                        view_direction=event.view_direction,
                        vector=selected_translator_normal,
                        dims_displayed=event.dims_displayed
                    )
                    translator_drag_vector = projected_distance * selected_translator_normal
                    self.centroid = self._initial_centroid + translator_drag_vector
                    self._while_translator_drag(selected_translator=selected_translator,
                                                translation_vector=translator_drag_vector)

                elif selected_rotator is not None:
                    # calculate the rotation matrix for the rotator drag
                    rotator_drag_vector = coordinates - initial_position_world
                    plane_normal = self.rotator_normals[selected_rotator]
                    projected_click_point, _ = project_points_onto_plane(
                        points=coordinates,
                        plane_point=self.centroid,
                        plane_normal=plane_normal,
                    )
                    click_vector = np.squeeze(projected_click_point) - self.centroid
                    rotation_matrix = rotation_matrix = rotation_matrix_from_vectors(
                        self._initial_click_vector, click_vector
                    )

                    # update the rotation matrix and call the _while_rotator_drag callback
                    self.rot_mat = np.dot(rotation_matrix, self._initial_rot_mat)
                    self._while_rotator_drag(
                        selected_rotator=selected_rotator,
                        rotation_matrix=rotation_matrix
                    )

                yield

        layer.interactive = True

        # Call a function to clean up after the mouse event
        self._initial_click_vector = None
        self._on_click_cleanup()

    def _setup_translator_drag(self, click_point: np.ndarray, selected_translator: Optional[int]):
        if selected_translator is not None:
            self._initial_centroid = self.centroid.copy()

    def _setup_rotator_drag(self, click_point: np.ndarray, selected_rotator: Optional[int]):
        if selected_rotator is not None:
            normal = self.rotator_normals[selected_rotator]

            # project the click point on to the plane of the rotat
            initial_click_point, _ = project_points_onto_plane(
                points=click_point,
                plane_point=self.centroid,
                plane_normal=normal,
            )

            self._initial_click_vector = np.squeeze(initial_click_point) - self.centroid
            self._initial_rot_mat = self.rot_mat.copy()

    def _pre_drag(
            self,
            click_point: np.ndarray,
            selected_translator: Optional[int],
            selected_rotator: Optional[int]
    ):
        """This is called at the beginning of the drag event and is
        typically used to save information that will be used during the
        drag callbacks (e.g., initial positions).

        Parameters
        ----------
        click_point : np.ndarray
            The click point in data coordinates (displayed dims only).
        selected_translator : Optional[int]
            The index of the selected translator. The index corresponds
            to self.translator_normals. This is None when
            no translator has been selected.
        selected_rotator : Optional[int]
            The index of the selected rotator. The index corresponds
            to self.rotator_normals. This is None when
            no rotator has been selected.
        """
        pass

    @abstractmethod
    def _while_translator_drag(self, selected_translator: int, translation_vector: np.ndarray):
        """This callback is called during translator drags events.

        Parameters
        ----------
        selected_translator : int
            The index of the selected translator. This index corresponds to
            the self.translator_normals.
        translation_vector : np.ndarray
            The vector describing the current dragposition from the initial position
            projected onto the selected translator.
        """
        pass

    def _while_rotator_drag(self, selected_rotator: int, rotation_matrix: np.ndarray):
        """This callback is called during rotator drag events.

        Parameters
        ----------
        selected_rotator : int
            The index of the selected rotator. This index corresponds to
            the self.rotator_normals.
        rotation_matrix : np.ndarray
            The (3 x 3) rotation matrix for the rotation from the initial point
            to the current point in the drag.
        """
        pass

    def _on_click_cleanup(self):
        """This callback is called at the end of the drag event and should
        be used to clean up any variables set during the click event.
        """
        pass

    def _set_canvas_none(self):
        self.node._set_canvas(None)

    def _on_visible_change(self):
        """Change visibiliy of axes."""
        self.node.visible = True
        self._on_zoom_change()

    def _update_translator_mesh(self):
        """Update the mesh for the translator"""
        translator_vertices, translator_indices, translator_colors, triangle_indices = make_translator_meshes(
            centroids=self.centroid,
            normals=self.translator_normals,
            colors=self._default_color[:len(self.translator_normals)],
            translator_length=self.translator_length,
            tube_radius=self._translator_width,
            tube_points=self._N_TUBE_POINTS,
        )
        self.translator_vertices = translator_vertices
        self.translator_indices = translator_indices
        self.translator_colors = translator_colors
        self.triangle_indices = triangle_indices
        self._on_data_change()

    def _update_rotator_mesh(self):
        """Update rotator mesh"""
        rotator_vertices, rotator_indices, rotator_colors, triangle_indices = make_rotator_meshes(
            centroids=np.repeat([0, 0, 0], 3, axis=0),
            normals=self._rotator_normals,
            colors=self._default_color[:len(self._initial_rotator_normals)],
            rotator_radius=self.rotator_radius,
            tube_radius=self.rotator_width,
            tube_points=self._N_TUBE_POINTS,
            n_segments=self._N_SEGMENTS_ROTATOR
        )

        self.rotator_vertices = rotator_vertices
        self.rotator_indices = rotator_indices
        self.rotator_colors = rotator_colors
        self.rotator_triangle_indices = triangle_indices

        self._on_matrix_change()


    def _on_data_change(self):
        """Change style of axes."""

        # Actual number of displayed dims
        ndisplay = self._viewer.dims.ndisplay

        # Determine data based the number of displayed dimensions
        if ndisplay == 2:
            # make the arrow lines
            data = self._line_data2D
            color = color_lines(self._default_color)
        elif ndisplay == 3:
            # make the arrow lines, reverse axes for vispy display
            translator_vertices = self.translator_vertices[:, ::-1]
            translator_indices = self.translator_indices.copy()
            translator_colors = self.translator_colors
        else:
            raise ValueError(
                trans._(
                    'Invalid ndisplay value',
                    deferred=True,
                )
            )

        if len(self._initial_rotator_normals) > 0:
            translator_indices += len(self.rotator_vertices)
            vertices = np.concatenate([self.rotator_vertices[:, ::-1], translator_vertices])
            faces = np.concatenate([self.rotator_indices, translator_indices])
            colors = np.concatenate([self.rotator_colors, translator_colors])
        else:
            vertices = translator_vertices
            faces = translator_indices
            colors = translator_colors

        self.node.set_data(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors
        )

    def _on_zoom_change(self):
        """Update axes length based on zoom scale."""
        # if not self._viewer.axes.visible:
        #     return
        return  # turn off temporarily
        scale = 1 / self._viewer.camera.zoom

        # If scale has not changed, do not redraw
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4:
            return
        self._scale = scale
        scale_canvas2world = self._scale
        target_canvas_pixels = self._target_length
        scale = target_canvas_pixels * scale_canvas2world
        # Update axes scale
        self.node.transform.scale = [scale, scale, scale, 1]


    def _on_matrix_change(self):
        # convert NumPy axis ordering to VisPy axis ordering
        # by reversing the axes order and flipping the linear
        # matrix
        translate = self.centroid[::-1]
        rot_matrix = self.rot_mat[::-1, ::-1].T

        # Embed in the top left corner of a 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[: rot_matrix.shape[0], : rot_matrix.shape[1]] = rot_matrix
        affine_matrix[-1, : len(translate)] = translate

        self.node.transform.matrix = affine_matrix
