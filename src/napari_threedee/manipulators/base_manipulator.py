from abc import ABC, abstractmethod

import numpy as np
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.theme import get_theme
from napari.utils.translations import trans
from napari_threedee.manipulators.manipulator_utils import select_rotator, color_lines
from napari_threedee.utils.selection_utils import select_line_segment, select_mesh_from_click
from vispy.scene import Compound, Line, Mesh, Text
from vispy.visuals.transforms import MatrixTransform


class BaseManipulator(ABC):
    _N_SEGMENTS_ROTATOR = 15
    def __init__(self, viewer, layer=None, order=0):
        super().__init__()
        self._viewer = viewer
        self._layer = layer

        self._layer.mouse_drag_callbacks.append(self.on_click)

        self._scale = 1
        self.rot_mat = np.eye(3)

        # Target axes length in canvas pixels
        self._target_length = 200
        # CMYRGB for 6 axes data in x, y, z, ... ordering
        self._default_color = [
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
        # Text offset from line end position
        self._text_offsets = 0.1 * np.array([1, 1, 1])

        # initialize the arrow lines
        self._init_arrow_lines()

        # initialize the rotators
        self._init_rotators()

        # get the layer node to pass as the parent to the visual
        visual = viewer.window.qt_viewer.layer_to_visual[layer]
        parent = visual._layer_node.get_node(3)
        self.node = Mesh(mode='triangles', shading='smooth', parent=parent)

        self.node.transform = MatrixTransform()
        self.node.order = order

        self.node.canvas._backend.destroyed.connect(self._set_canvas_none)

        self._viewer.camera.events.zoom.connect(self._on_zoom_change)

        # self._on_visible_change()
        self._on_matrix_change()
        self._on_data_change()
        self.node.visible = True

    def _init_arrow_lines(self):
        self._line_data2D = None
        self._line_data3D = None
        self.translator_vertices = None
        self._initial_translator_normals = np.empty()

    def _init_rotators(self):
        self._rotator_data2D = None
        self._rotator_data3D = None
        self._rotator_connections = None
        self.rotator_vertices = None
        self._initial_rotator_normals = np.empty()

    # @property
    # def translator_normals(self) -> np.ndarray:
    #     return (self._initial_translator_normals @ self.rot_mat.T)
    # @property
    # def rotator_normals(self) -> np.ndarray:
    #     return (self._initial_rotator_normals @ self.rot_mat.T)


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
        # plane_point = plane_point[::-1]

        view_dir_data = np.asarray(self._layer._world_to_data_ray(event.view_direction))
        plane_normal = view_dir_data[event.dims_displayed]
        # plane_normal = plane_normal[::-1]

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

        initial_position_world = event.position
        yield

        if selected_translator is not None or selected_rotator is not None:
            # set up for the mouse drag
            self._pre_drag(plane_point, selected_rotator)
        #
            while event.type == 'mouse_move':
                # click position
                coordinates = np.asarray(layer.world_to_data(event.position))[event.dims_displayed]

                rotator_drag_vector = None
                translator_drag_vector = None
                if selected_translator is not None:
                # get
                    projected_distance = layer.projected_distance_from_mouse_drag(
                        start_position=initial_position_world,
                        end_position=event.position,
                        view_direction=event.view_direction,
                        vector=selected_translator_normal,
                        dims_displayed=event.dims_displayed
                    )
                    translator_drag_vector = projected_distance * selected_translator_normal
                    self._while_translator_drag(translator_drag_vector)

                elif selected_rotator is not None:
                    rotator_drag_vector = coordinates - initial_position_world
                    self._while_rotator_drag(coordinates, rotator_drag_vector, selected_rotator)


                yield
        #
        layer.interactive = True
        #
        # # Call a function to clean up after the mouse event
        # self._on_click_cleanup()

    @abstractmethod
    def _pre_drag(self, click_point_data_displayed, rotator_index):
        raise NotImplementedError

    @abstractmethod
    def _while_translator_drag(self, translation_vector):
        raise NotImplementedError

    def _while_rotator_drag(self, click_position, rotation_drag_vector, rotator_selection):
        raise NotImplementedError

    @abstractmethod
    def _on_click_cleanup(self):
        raise NotImplementedError

    def _set_canvas_none(self):
        self.node._set_canvas(None)

    def _on_visible_change(self):
        """Change visibiliy of axes."""
        self.node.visible = True
        self._on_zoom_change()
        self._on_data_change()

    def _on_data_change(self):
        """Change style of axes."""
        # if not self._viewer.axes.visible:
        #     return

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

        translator_indices += len(self.rotator_vertices)
        vertices = np.concatenate([self.rotator_vertices[:, ::-1], translator_vertices])
        faces = np.concatenate([self.rotator_indices, translator_indices])
        colors = np.concatenate([self.rotator_colors, translator_colors])

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
