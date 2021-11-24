from abc import ABC, abstractmethod

import numpy as np
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.theme import get_theme
from napari.utils.translations import trans
from napari_threedee.manipulators.manipulator_utils import select_rotator, color_lines
from napari_threedee.utils.selection_utils import select_line_segment
from vispy.scene import Compound, Line, Text
from vispy.visuals.transforms import STTransform


class BaseManipulator(ABC):
    _N_SEGMENTS_ROTATOR = 100
    def __init__(self, viewer, layer=None, order=0):
        super().__init__()
        self._viewer = viewer
        self._layer = layer

        self._layer.mouse_drag_callbacks.append(self.on_click)

        self._scale = 1

        # Target axes length in canvas pixels
        self._target_length = 200
        # CMYRGB for 6 axes data in x, y, z, ... ordering
        self._default_color = [
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
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
        self.node = Compound(
            [Line(connect='segments', method='gl', width=3), Line(connect='segments', method='gl', width=3), Text()],
            parent=parent,
        )

        self.node.transform = STTransform(translate=(0, 0, 0), scale=(1, 1, 1, 1))
        self.node.order = order

        # Add a text node to display axes labels
        self.text_node = self.node._subvisuals[2]
        self.text_node.font_size = 10
        self.text_node.anchors = ('center', 'center')
        self.text_node.text = f'{1}'

        # Access the line ode
        self.line_node = self.node._subvisuals[0]

        self.node.canvas._backend.destroyed.connect(self._set_canvas_none)

        self._viewer.camera.events.zoom.connect(self._on_zoom_change)

        # self._on_visible_change()
        self._on_data_change()
        self.node.visible = True

    def _init_arrow_lines(self):
        self._line_data2D = None
        self._line_data3D = None

    def _init_rotators(self):
        self._rotator_data2D = None
        self._rotator_data3D = None
        self._rotator_connections = None

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
        if self._line_data3D is not None:
            line_segment_points = self._line_data3D
            potential_matches = select_line_segment(
                line_segment_points=line_segment_points,
                plane_normal=plane_normal,
                plane_point=plane_point,
                distance_threshold=3
            )
        else:
            potential_matches = []

        if self._rotator_data3D is not None:
            rotator_selection = select_rotator(
                click_position=point_data,
                plane_normal=plane_normal,
                rotator_data=self._rotator_data3D
            )
            if rotator_selection is not None:
                rotator_index = self._rotator_vertex_axis[rotator_selection]
                print(rotator_index)
            else:
                rotator_index = None
        else:
            rotator_index = None

        if len(potential_matches) == 1:
            layer.interactive = False
            match = potential_matches[0]

            match_vector = line_segment_points[2 * match + 1, :] - line_segment_points[2 * match, :]
            print(line_segment_points, match_vector)
            match_vector = match_vector / np.linalg.norm(match_vector)
        elif rotator_index is not None:
            layer.interactive = False
            match = None
            match_vector = None
        else:
            match = None
            match_vector = None
        initial_position_world = event.position
        yield

        if match is not None or rotator_index is not None:
            # set up for the mouse drag
            self._pre_drag(plane_point, rotator_index)

            while event.type == 'mouse_move':
                # click position
                coordinates = np.asarray(layer.world_to_data(event.position))[event.dims_displayed]

                rotator_drag_vector = None
                translator_drag_vector = None
                if match is not None:
                # get
                    projected_distance = layer.projected_distance_from_mouse_drag(
                        start_position=initial_position_world,
                        end_position=event.position,
                        view_direction=event.view_direction,
                        vector=match_vector,
                        dims_displayed=event.dims_displayed
                    )
                    translator_drag_vector = projected_distance * match_vector
                    self._while_translator_drag(translator_drag_vector, rotator_drag_vector, rotator_selection)

                elif rotator_selection is not None:
                    rotator_drag_vector = coordinates - initial_position_world
                    self._while_rotator_drag(coordinates, rotator_drag_vector, rotator_index)


                yield

        layer.interactive = True

        # Call a function to clean up after the mouse event
        self._on_click_cleanup()

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
        self.text_node._set_canvas(None)

    def _on_visible_change(self):
        """Change visibiliy of axes."""
        self.node.visible = True
        self._on_zoom_change()
        self._on_data_change()

    def _on_data_change(self):
        """Change style of axes."""
        # if not self._viewer.axes.visible:
        #     return

        # Determine which axes are displayed
        axes = self._viewer.dims.displayed

        # Actual number of displayed dims
        ndisplay = self._viewer.dims.ndisplay

        # Determine the labels of those axes
        axes_labels = [self._viewer.dims.axis_labels[a] for a in axes[::-1]]
        # Counting backwards from total number of dimensions
        # determine axes positions. This is done as by default
        # the last NumPy axis corresponds to the first Vispy axis
        reversed_axes = [self._viewer.dims.ndim - 1 - a for a in axes[::-1]]

        # Determine colors of axes based on reverse position
        if self._viewer.axes.colored:
            axes_colors = [
                self._default_color[ra % len(self._default_color)]
                for ra in reversed_axes
            ]
        else:
            # the reason for using the `as_hex` here is to avoid
            # `UserWarning` which is emitted when RGB values are above 1
            background_color = get_theme(
                self._viewer.theme, False
            ).canvas.as_hex()
            background_color = transform_color(background_color)[0]
            color = np.subtract(1, background_color)
            color[-1] = background_color[-1]
            axes_colors = [color] * ndisplay

        # Determine data based the number of displayed dimensions
        if ndisplay == 2:

            # make the arrow lines
            data = self._line_data2D
            color = color_lines(axes_colors)
        elif ndisplay == 3:
            # make the arrow lines
            data = self._line_data3D[:, ::-1]
            n_axes = int(data.shape[0] / 2)
            color = color_lines(axes_colors[:n_axes])
        else:
            raise ValueError(
                trans._(
                    'Invalid ndisplay value',
                    deferred=True,
                )
            )

        # set the lines data
        self.node._subvisuals[0].set_data(data, color)

        if self._rotator_data3D is not None:
            rotator_data = self._rotator_data3D[:, ::-1]
            rotator_colors = np.concatenate(
                [
                    [axes_colors[0]] * self._N_SEGMENTS_ROTATOR,
                    [axes_colors[1]] * self._N_SEGMENTS_ROTATOR,
                    [axes_colors[2]] * self._N_SEGMENTS_ROTATOR
                ],
                axis=0,
            )
            self.node._subvisuals[1].set_data(rotator_data, rotator_colors, connect=self._rotator_connections)

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
