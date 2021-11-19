import numpy as np
from vispy.scene.visuals import Compound, Line, Mesh, Text
from vispy.visuals.transforms import STTransform

from napari.layers.shapes._shapes_utils import triangulate_ellipse
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.theme import get_theme
from napari.utils.translations import trans
from napari.utils.geometry import project_points_onto_plane, rotate_points


def make_dashed_line(num_dashes, axis):
    """Make a dashed line.

    Parameters
    ----------
    num_dashes : int
        Number of dashes in the line.
    axis : int
        Axis which is dashed.

    Returns
    -------
    np.ndarray
        Dashed line, of shape (num_dashes, 3) with zeros in
        the non dashed axes and line segments in the dashed
        axis.
    """
    dashes = np.linspace(0, 1, num_dashes * 2)
    dashed_line_ends = np.concatenate(
        [[dashes[2 * i], dashes[2 * i + 1]] for i in range(num_dashes)], axis=0
    )
    dashed_line = np.zeros((2 * num_dashes, 3))
    dashed_line[:, axis] = np.array(dashed_line_ends)
    return dashed_line


def make_arrow_head(num_segments, axis):
    """Make an arrowhead line.

    Parameters
    ----------
    num_segments : int
        Number of segments in the arrowhead.
    axis
        Arrowhead direction.

    Returns
    -------
    np.ndarray, np.ndarray
        Vertices and faces of the arrowhead.
    """
    corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.1
    vertices, faces = triangulate_ellipse(corners, num_segments)
    full_vertices = np.zeros((num_segments + 1, 3))
    inds = list(range(3))
    inds.pop(axis)
    full_vertices[:, inds] = vertices
    full_vertices[:, axis] = 0.9
    full_vertices[0, axis] = 1.02
    return full_vertices, faces


def color_lines(colors):
    if len(colors) == 2:
        return np.concatenate(
            [[colors[0]] * 2, [colors[1]] * 2],
            axis=0,
        )
    elif len(colors) == 3:
        return np.concatenate(
            [[colors[0]] * 2, [colors[1]] * 2, [colors[2]] * 2],
            axis=0,
        )
    else:
        return ValueError(
            trans._(
                'Either 2 or 3 colors must be provided, got {number}.',
                deferred=True,
                number=len(colors),
            )
        )


def color_dashed_lines(colors):
    if len(colors) == 2:
        return np.concatenate(
            [[colors[0]] * 2, [colors[1]] * 4 * 2],
            axis=0,
        )
    elif len(colors) == 3:
        return np.concatenate(
            [[colors[0]] * 2, [colors[1]] * 4 * 2, [colors[2]] * 8 * 2],
            axis=0,
        )
    else:
        return ValueError(
            trans._(
                'Either 2 or 3 colors must be provided, got {number}.',
                deferred=True,
                number=len(colors),
            )
        )


def color_arrowheads(colors, num_segments):
    if len(colors) == 2:
        return np.concatenate(
            [[colors[0]] * num_segments, [colors[1]] * num_segments],
            axis=0,
        )
    elif len(colors) == 3:
        return np.concatenate(
            [
                [colors[0]] * num_segments,
                [colors[1]] * num_segments,
                [colors[2]] * num_segments,
            ],
            axis=0,
        )
    else:
        return ValueError(
            trans._(
                'Either 2 or 3 colors must be provided, got {number}.',
                deferred=True,
                number=len(colors),
            )
        )

def distance_between_point_and_line_segment_2d(p, p1, p2):
    """Calculate the distance between a point p and a line segment p1, p2
    """
    x0 = p[0]
    y0 = p[1]
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    numerator = np.linalg.norm((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))
    denominator = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    return numerator / denominator



class VispyAxesOverlay:
    """Axes indicating world coordinate origin and orientation."""

    _NUM_SEGMENTS_ARROWHEAD = 100

    def __init__(self, viewer, layer=None, order=0):
        self._viewer = viewer
        self._layer = layer

        self._layer.mouse_drag_callbacks.append(self.on_click)

        self._scale = 1

        # Target axes length in canvas pixels
        self._target_length = 200
        # CMYRGB for 6 axes data in x, y, z, ... ordering
        self._default_color = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
        # Text offset from line end position
        self._text_offsets = 0.1 * np.array([1, 1, 1])

        # note order is x, y, z for VisPy
        self._line_data2D = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]]
        )
        self._line_data3D = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
        )

        # note order is x, y, z for VisPy
        self._dashed_line_data2D = np.concatenate(
            [[[1, 0, 0], [0, 0, 0]], make_dashed_line(4, axis=1)],
            axis=0,
        )
        self._dashed_line_data3D = np.concatenate(
            [
                [[1, 0, 0], [0, 0, 0]],
                make_dashed_line(4, axis=1),
                make_dashed_line(8, axis=2),
            ],
            axis=0,
        )

        # note order is x, y, z for VisPy
        vertices = np.empty((0, 3))
        faces = np.empty((0, 3))
        for axis in range(2):
            v, f = make_arrow_head(self._NUM_SEGMENTS_ARROWHEAD, axis)
            faces = np.concatenate([faces, f + len(vertices)], axis=0)
            vertices = np.concatenate([vertices, v], axis=0)
        self._default_arrow_vertices2D = vertices
        self._default_arrow_faces2D = faces.astype(int)

        vertices = np.empty((0, 3))
        faces = np.empty((0, 3))
        for axis in range(3):
            v, f = make_arrow_head(self._NUM_SEGMENTS_ARROWHEAD, axis)
            faces = np.concatenate([faces, f + len(vertices)], axis=0)
            vertices = np.concatenate([vertices, v], axis=0)
        self._default_arrow_vertices3D = vertices
        self._default_arrow_faces3D = faces.astype(int)

        # get the layer node to pass as the parent to the visual
        visual = viewer.window.qt_viewer.layer_to_visual[layer]
        parent = visual._layer_node.get_node(3)
        self.node = Compound(
            [Line(connect='segments', method='gl', width=3), Mesh(), Text()],
            parent=parent,
        )

        self.node.transform = STTransform(translate=(0, 50, 50), scale=(50, 50, 50, 1))
        self.node.order = order

        # Add a text node to display axes labels
        self.text_node = self.node._subvisuals[2]
        self.text_node.font_size = 10
        self.text_node.anchors = ('center', 'center')
        self.text_node.text = f'{1}'

        # Access the line ode
        self.line_node = self.node._subvisuals[0]
        # self.line_node.set_data(width=50)

        self.node.canvas._backend.destroyed.connect(self._set_canvas_none)
        # End Note

        # self._viewer.events.theme.connect(self._on_data_change)
        # self._viewer.axes.events.visible.connect(self._on_visible_change)
        # self._viewer.axes.events.colored.connect(self._on_data_change)
        # self._viewer.axes.events.dashed.connect(self._on_data_change)
        # self._viewer.axes.events.labels.connect(self._on_data_change)
        # self._viewer.axes.events.arrows.connect(self._on_data_change)
        # self._viewer.dims.events.order.connect(self._on_data_change)
        # self._viewer.dims.events.range.connect(self._on_data_change)
        # self._viewer.dims.events.ndisplay.connect(self._on_data_change)
        # self._viewer.dims.events.axis_labels.connect(self._on_data_change)
        self._viewer.camera.events.zoom.connect(self._on_zoom_change)

        # self._on_visible_change()
        self._on_data_change()
        self.node.visible = True

    def on_click(self, layer, event):
        """Mouse call back for selecting and dragging an axis"""

        # get the points and vectors in data coordinates
        point_world = event.position
        point_data = np.asarray(self._layer.world_to_data(point_world))
        plane_point = point_data[event.dims_displayed]
        plane_point = plane_point[::-1]

        view_dir_data = np.asarray(self._layer._world_to_data_ray(event.view_direction))
        plane_normal = view_dir_data[event.dims_displayed]
        plane_normal = plane_normal[::-1]

        # project the in view points onto the plane
        line_segment_points = (50 * self._line_data3D) + np.array([0, 50, 50])
        projected_points, projection_distances = project_points_onto_plane(
            points=line_segment_points,
            plane_point=plane_point,
            plane_normal=plane_normal,
        )

        # rotate points and plane to be axis aligned with normal [0, 0, 1]
        rotated_points, rotation_matrix = rotate_points(
            points=projected_points,
            current_plane_normal=plane_normal,
            new_plane_normal=[0, 0, 1],
        )
        rotated_click_point = np.dot(rotation_matrix, plane_point)

        rotated_points_2d = rotated_points[:, :2]
        rotated_click_point_2d = rotated_click_point[:2]

        # get distance between click and projected axes
        distances = []
        for i in range(3):
            p_0 = rotated_points_2d[i * 2]
            p_1 = rotated_points_2d[i * 2 + 1]
            dist = distance_between_point_and_line_segment_2d(rotated_click_point_2d, p_0, p_1)
            distances.append(dist)
        distances = np.asarray(distances)
        # determine if any of the axes were clicked based on their width
        potential_matches = np.argwhere(distances < 10)
        print(distances, potential_matches)
        if len(potential_matches) == 1:
            layer.interactive = False
            match = potential_matches[0]
            if match == 0:
                match = 2
            elif match == 2:
                match = 0
            match_vector = line_segment_points[2 * match + 1, ::-1]
            match_vector = match_vector / np.linalg.norm(match_vector)
        else:
            match = None
            match_vector = None
        last_position = event.position
        yield

        if match is not None:

            while event.type == 'mouse_move':
                # click position
                coordinates = np.asarray(layer.world_to_data(event.position))[event.dims_displayed]

                # get
                projected_distance = layer.projected_distance_from_mouse_drag(
                    start_position=last_position,
                    end_position=event.position,
                    view_direction=event.view_direction,
                    vector=match_vector,
                    dims_displayed=event.dims_displayed
                )
                current_translation = layer.translate
                translation_increment = np.zeros((3,))
                translation_increment[match] = projected_distance
                new_translation = current_translation + translation_increment
                layer.translate = new_translation

                # update the last click position
                last_position = event.position
                yield

        layer.interactive = True

    # @property
    # def thickness(self):
    #     return self._thickness
    #
    # @thickness.setter
    # def thickness(self, value: float):
    #     self._thickness = value
    #     # self.node.

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
        ndisplay = len(self._viewer.dims.displayed)

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

        # Determine data based on number of displayed dimensions and
        # axes visualization parameters
        if self._viewer.axes.dashed and ndisplay == 2:
            data = self._dashed_line_data2D
            color = color_dashed_lines(axes_colors)
            text_data = self._line_data2D[1::2]
        elif self._viewer.axes.dashed and ndisplay == 3:
            data = self._dashed_line_data3D
            color = color_dashed_lines(axes_colors)
            text_data = self._line_data3D[1::2]
        elif not self._viewer.axes.dashed and ndisplay == 2:
            data = self._line_data2D
            color = color_lines(axes_colors)
            text_data = self._line_data2D[1::2]
        elif not self._viewer.axes.dashed and ndisplay == 3:
            data = self._line_data3D
            color = color_lines(axes_colors)
            text_data = self._line_data3D[1::2]
        else:
            raise ValueError(
                trans._(
                    'Axes dash status and ndisplay combination not supported',
                    deferred=True,
                )
            )

        if self._viewer.axes.arrows and ndisplay == 2:
            arrow_vertices = self._default_arrow_vertices2D
            arrow_faces = self._default_arrow_faces2D
            arrow_color = color_arrowheads(
                axes_colors, self._NUM_SEGMENTS_ARROWHEAD
            )
        elif self._viewer.axes.arrows and ndisplay == 3:
            arrow_vertices = self._default_arrow_vertices3D
            arrow_faces = self._default_arrow_faces3D
            arrow_color = color_arrowheads(
                axes_colors, self._NUM_SEGMENTS_ARROWHEAD
            )
        else:
            arrow_vertices = np.zeros((3, 3))
            arrow_faces = np.array([[0, 1, 2]])
            arrow_color = [[0, 0, 0, 0]]

        self.node._subvisuals[0].set_data(data, color)
        self.node._subvisuals[1].set_data(
            vertices=arrow_vertices,
            faces=arrow_faces,
            face_colors=arrow_color,
        )

        # Set visibility status of text
        self.text_node.visible = (
            self._viewer.axes.visible and self._viewer.axes.labels
        )
        self.text_node.text = axes_labels
        self.text_node.color = axes_colors
        self.text_node.pos = text_data + self._text_offsets

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
