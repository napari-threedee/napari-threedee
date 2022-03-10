from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type

import napari
import numpy as np
from napari.utils.geometry import project_points_onto_plane, rotation_matrix_from_vectors_3d
from napari.utils.translations import trans
from napari.viewer import Viewer
from vispy.scene import Mesh
from vispy.visuals.transforms import MatrixTransform

from .manipulator_utils import make_translator_meshes, color_lines, make_rotator_meshes
from ..base import ThreeDeeModel
from ..utils.napari_utils import get_vispy_node, add_mouse_callback_safe, remove_mouse_callback_safe
from ..utils.selection_utils import select_mesh_from_click


class BaseManipulator(ThreeDeeModel, ABC):
    """Base class for manipulators.

    To implement:
        - the __init__() should take the viewer as the first argument, the layer
            as the second argument followed by any keyword arguments.
            Keyword arguments should have default values.
        - implement a self._set_initial_translation_vectors() method
        - implement a self._set_initial_rotator_normals() method
        - implement the _pre_drag() callback.
        - implement the _while_dragging_translator() callback.
        - implement the _while_dragging_rotator() callback.
        - implement the _post_drag() callback.
        - Call the super.__init__().

    Parameters
    ----------
    viewer : Viewer
        The napari viewer containing the visuals.
    layer : Optional[Layer]
        The callback_list to attach the manipulator to.
    order : int
        The order to place the manipulator visuals in the vispy scene graph.
    translator_length : float
        The length of the translator arms in data units.
    translator_width : float
        The width of the translator arms in data units.
    rotator_radius : float
        The radius of the rotators in data units.
    rotator_width : float
        The width of the rotators in data units.

    Attributes
    ----------
    translation : np.ndarray
        (3, 1) array containing the coordinates to the translation of the manipulator.
    rot_mat : np.ndarray
        (3, 3) array containing the rotation matrix applied to the manipulator.
    translator_length : float
        The length of the translator arms in data units.
    translator_width : float
        The width of the translator arms in data units.
    rotator_radius : float
        The radius of the rotators in data units.
    rotator_width : float
        The width of the rotators in data units.
    translator_normals : np.ndarray
        (N x 3) array containing the normal vector for each of the N translators.
    rotator_normals : np.ndarray
        (N x 3) array containing the normal vector for each of the N rotators.

    Notes
    -----
    _N_SEGMENTS_ROTATOR : float
        The number of segments to discretize the rotator into. More segments
        makes the rotator look more smooth, but will reduce rendering performance.
    _N_TUBE_POINTS : float
        The number of points to use to represent the circular crossection of the
        manipulator objects. More points makes the manipulator appear more smooth, but
        will reduce the rendering performance.
    """
    _N_SEGMENTS_ROTATOR = 50
    _N_TUBE_POINTS = 15

    def __init__(
            self,
            viewer,
            layer=None,
            enabled: bool = True,
            order: int = 0,
            translator_length: float = 50,
            translator_width: float = 1,
            rotator_radius: float = 5,
            rotator_width: float = 1,
    ):
        super().__init__()
        self._viewer = viewer

        # this will be overriden by the input layer below
        self._layer = None
        self._enabled = False

        self._translator_length = translator_length
        self._translator_width = translator_width
        self._rotator_radius = rotator_radius
        self._rotator_width = rotator_width

        # this is used to store the vector to the initial click
        # on a rotator for calculating the rotation
        self._initial_click_vector = None

        # this is used to store the initial rotation matrix before
        # starting a rotation
        self._initial_rot_mat = None

        # CMYRGB for 6 axes data in x, y, z, ... ordering
        self._default_color = [
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]

        # the order for the manipulator visual in the vispy scenegraph
        self._vispy_visual_order = order

        # connect events
        self._viewer.camera.events.zoom.connect(self._on_zoom_change)
        self._viewer.dims.events.ndisplay.connect(self._on_ndisplay_change)
        # add the layer
        self.layer = layer

        self.enabled = enabled

    @abstractmethod
    def _initialize_transform(self):
        """Should set the properties required to calculate the rotation matrix and translation"""
        raise NotImplementedError

    @property
    def _initial_translation_vectors(self):
        """An (Nx3) numpy array containing the translation vector for each of the
        N translators to be created.

        Translation vectors are defined in displayed data coordinates.
        """
        return self._initial_translation_vectors_

    @_initial_translation_vectors.setter
    def _initial_translation_vectors(self, value: np.ndarray):
        """An (Nx3) numpy array containing the translation vector for each of the
        N translators to be created.

        Translation vectors are defined in displayed data coordinates.
        """
        value = np.asarray(value)
        self._initial_translation_vectors_ = value

    def _set_initial_translation_vectors(self):
        """Method to be overridden by subclasses for defining translation vectors.
        """
        self._initial_translation_vectors = np.empty((0, 3))

    def _init_translators(self):
        translator_vertices, translator_indices, translator_colors, triangle_indices = make_translator_meshes(
            centroids=np.asarray([0, 0, 0]),
            normals=self._initial_translation_vectors,
            colors=self._default_color[:len(self._initial_translation_vectors)],
            translator_length=self.translator_length,
            tube_radius=self._translator_width,
            tube_points=self._N_TUBE_POINTS,
        )

        self.translator_vertices = translator_vertices
        self.translator_indices = translator_indices
        self.translator_colors = translator_colors
        self.translator_triangle_indices = triangle_indices

        self._translator_normals = self._initial_translation_vectors.copy()

    @property
    def _initial_rotator_normals(self):
        """An (Nx3) numpy array containing the normal direction for each of the
        N rotators to be created.

        Normal directions are defined in displayed data coordinates.
        """
        return self._initial_rotator_normals_

    @_initial_rotator_normals.setter
    def _initial_rotator_normals(self, value: np.ndarray):
        """An (Nx3) numpy array containing the normal direction for each of the
        N rotators to be created.

        Normal directions are defined in displayed data coordinates.
        """
        value = np.asarray(value)
        self._initial_rotator_normals_ = value

    def _set_initial_rotator_normals(self):
        """Method to be overridden by subclasses for defining rotator normals.
        """
        self._initial_rotator_normals = np.empty((0, 3))

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
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer: Optional[Type[napari.layers.Layer]]):
        if layer is not None:
            if layer == self._layer:
                return
            if self.layer is not None:
                # remove the current visual
                self.node.parent = None
                self._disconnect_events(self.layer)
            self._layer = layer
            self._connect_vispy_visual(layer)
            self._initialize_layer()

            if self.enabled:
                self._on_enable()
            else:
                self._on_disable()
            self._connect_events(self.layer)
        else:
            self._layer = layer

    def _initialize_layer(self):
        self._initialize_transform()
        self._set_initial_translation_vectors()
        self._init_translators()
        self._set_initial_rotator_normals()
        self._init_rotators()
        self._on_data_change()
        self._on_zoom_change()
        self._on_matrix_change()

    def _connect_vispy_visual(self, layer: Type[napari.layers.Layer]):
        # get the callback_list node to pass as the parent of the manipulator
        parent = get_vispy_node(self._viewer, layer)
        self.node = Mesh(mode='triangles', shading='smooth', parent=parent)

        self.node.transform = MatrixTransform()
        self.node.order = self._vispy_visual_order
        self.node.canvas._backend.destroyed.connect(self._set_canvas_none)

    def _connect_events(self, layer: napari.layers.Layer):
        """This method should be implemented on subclasses that
        require events to be connected to the layer when self.layer
        is set (other than the main mouse callback"""
        pass

    def _disconnect_events(self, layer: napari.layers.Layer):
        """This method must be implemented on subclasses that
        implement _connect_events(). This methood is to disconnect
        the events that were connected in _connect_events()"""
        pass


    def set_layers(self, layer: Type[napari.layers.Layer]):
        """Override this in a subclass with the correct layer type for the manipulator"""
        self.layer = layer

    def _on_enable(self):
        if self.layer is not None:
            self.node.visible = True
            add_mouse_callback_safe(
                self._layer.mouse_drag_callbacks,
                self._mouse_callback,
                index=0
            )

    def _on_disable(self):
        if self.layer is not None:
            self.node.visible = False
            remove_mouse_callback_safe(
                self._layer.mouse_drag_callbacks,
                self._mouse_callback
            )

    def _on_ndisplay_change(self, event=None):
        if self._viewer.dims.ndisplay == 2:
            self.enabled = False
            self._disconnect_events(self.layer)
        else:
            self.enabled = True
            self._connect_events(self.layer)

    @property
    def translation(self) -> np.ndarray:
        """Vector from the layer origin to the manipulator origin"""
        return self._translation

    @translation.setter
    def translation(self, translation: np.ndarray):
        self._translation = np.asarray(translation)
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
        return (self._initial_translation_vectors @ self.rot_mat.T)

    @property
    def rotator_normals(self) -> np.ndarray:
        return (self._initial_rotator_normals @ self.rot_mat.T)

    @property
    def _displayed_translator_vertices(self):
        if self.translator_vertices is not None:
            return (self.translator_vertices @ self.rot_mat.T) + self.translation
        else:
            return None

    @property
    def _displayed_rotator_vertices(self):
        if self.rotator_vertices is not None:
            return (self.rotator_vertices @ self.rot_mat.T) + self.translation
        else:
            return None

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
        if selected_translator is not None:
            selected_translator_normal = self.translator_normals[selected_translator]
        else:
            selected_translator_normal = None
        if (selected_rotator is not None) or (selected_translator is not None):
            layer.interactive = False

        initial_position_world = event.position
        yield

        if selected_translator is not None or selected_rotator is not None:

            self._setup_translator_drag(
                click_point=click_position_data_3d, selected_translator=selected_translator
            )
            # set up for the mouse drag
            self._setup_rotator_drag(
                click_point=click_position_data_3d, selected_rotator=selected_rotator
            )

            # call the _pre_drag callback
            self._pre_drag(
                click_point=click_position_data_3d,
                selected_translator=selected_translator,
                selected_rotator=selected_rotator
            )

            while event.type == 'mouse_move':
                # click position
                coordinates = np.asarray(layer.world_to_data(event.position))[event.dims_displayed]

                # rotator_drag_vector = None
                # translator_drag_vector = None
                self._process_translator_drag(
                    event,
                    selected_translator,
                    initial_position_world,
                    selected_translator_normal
                )
                self._process_rotator_drag(
                    event,
                    selected_rotator,
                    coordinates,
                    initial_position_world
                )

                yield

        # Call a function to clean up after the mouse event
        layer.interactive = initial_layer_interactive
        self._initial_click_vector = None
        self._initial_rot_mat = None
        self._layer._drag_start = None
        self._post_drag()

    def _process_translator_drag(
            self,
            event,
            selected_translator: int,
            initial_position_world: np.ndarray,
            selected_translator_normal: np.ndarray
    ):
        if selected_translator is None:
            # no processing necessary if a translator was not selected
            return
        # get drag vector projected onto the translator axis
        projected_distance = self.layer.projected_distance_from_mouse_drag(
            start_position=initial_position_world,
            end_position=event.position,
            view_direction=event.view_direction,
            vector=selected_translator_normal,
            dims_displayed=event.dims_displayed
        )
        translator_drag_vector = projected_distance * selected_translator_normal
        self.translation = self._initial_translation + translator_drag_vector
        self._while_dragging_translator(selected_translator=selected_translator,
                                        translation_vector=translator_drag_vector)

    def _process_rotator_drag(
            self,
            event,
            selected_rotator: int,
            coordinates,
            initial_position_world
    ):
        if selected_rotator is None:
            # no processing necessary if a rotator was not selected
            return
        # calculate the rotation matrix for the rotator drag
        rotator_drag_vector = coordinates - initial_position_world
        plane_normal = self.rotator_normals[selected_rotator]
        projected_click_point, _ = project_points_onto_plane(
            points=coordinates,
            plane_point=self.translation,
            plane_normal=plane_normal,
        )
        click_vector = np.squeeze(projected_click_point) - self.translation
        rotation_matrix = rotation_matrix_from_vectors_3d(
            self._initial_click_vector, click_vector
        )

        # update the rotation matrix and call the _while_rotator_drag callback
        self.rot_mat = np.dot(rotation_matrix, self._initial_rot_mat)
        self._while_dragging_rotator(
            selected_rotator=selected_rotator,
            rotation_matrix=rotation_matrix
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
        if len(self.translator_normals) > 0:
            translator_triangles = self._displayed_translator_vertices[self.translator_indices]
            selected_translator = select_mesh_from_click(
                click_point=plane_point,
                view_direction=plane_normal,
                triangles=translator_triangles,
                triangle_indices=self.translator_triangle_indices
            )
        else:
            selected_translator = None

        if len(self.rotator_normals) > 0:
            rotator_triangles = self._displayed_rotator_vertices[self.rotator_indices]
            selected_rotator = select_mesh_from_click(
                click_point=plane_point,
                view_direction=plane_normal,
                triangles=rotator_triangles,
                triangle_indices=self.rotator_triangle_indices
            )
        else:
            selected_rotator = None

        return selected_translator, selected_rotator

    def _setup_translator_drag(self, click_point: np.ndarray, selected_translator: Optional[int]):
        if selected_translator is not None:
            self._initial_translation = self.translation.copy()

    def _setup_rotator_drag(self, click_point: np.ndarray, selected_rotator: Optional[int]):
        if selected_rotator is not None:
            normal = self.rotator_normals[selected_rotator]

            # project the click point on to the plane of the rotat
            initial_click_point, _ = project_points_onto_plane(
                points=click_point,
                plane_point=self.translation,
                plane_normal=normal,
            )

            self._initial_click_vector = np.squeeze(initial_click_point) - self.translation
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

    def _while_dragging_translator(self, selected_translator: int, translation_vector: np.ndarray):
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

    def _while_dragging_rotator(self, selected_rotator: int, rotation_matrix: np.ndarray):
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

    def _post_drag(self):
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
            centroids=self.translation,
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
        if self.layer is None:
            # do not do anything if the layer has not been set
            return

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
        # if not self._viewer.axes.enabled:
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
        if self.layer is None:
            # do not do anything if the layer has not been set
            return
        # convert NumPy axis ordering to VisPy axis ordering
        # by reversing the axes order and flipping the linear
        # matrix
        translate = self.translation[::-1]
        rot_matrix = self.rot_mat[::-1, ::-1].T

        # Embed in the top left corner of a 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[: rot_matrix.shape[0], : rot_matrix.shape[1]] = rot_matrix
        affine_matrix[-1, : len(translate)] = translate

        self.node.transform.matrix = affine_matrix
