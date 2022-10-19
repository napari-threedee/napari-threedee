from typing import Tuple

import numpy as np
from napari.utils.geometry import project_points_onto_plane, rotate_points, rotation_matrix_from_vectors_3d
from napari.utils.translations import trans
from vispy.util.transforms import rotate

import collections

def color_lines(colors):
    if len(colors) == 1:
        return np.concatenate(
            [[colors[0]] * 2],
            axis=0,
        )

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
                'Either 1, 2 or 3 colors must be provided, got {number}.',
                deferred=True,
                number=len(colors),
            )
        )


def create_circle_line_segments(centroid: np.ndarray, normal: np.ndarray, radius: float, n_segments: int) -> np.ndarray:

    # create the line segments for a unit circle
    # centered around [0, 0, 0] with normal [0, 0, 1]
    line_segments = np.zeros((n_segments, 3))

    angles = np.linspace(0, 2 * np.pi, n_segments)
    for i in range(n_segments):
        # todo numpify
        start_angle = angles[i]
        line_segments[i, ...] = [np.cos(start_angle), np.sin(start_angle), 0]

    # set the radius
    line_segments *= radius

    # seg the angle
    line_segments, _ = rotate_points(
        points=line_segments,
        current_plane_normal=[0, 0, 1],
        new_plane_normal=normal,
    )

    # set the center
    line_segments += centroid


    return line_segments


def create_axis_line_segment(
        centroid: np.ndarray,
        normal: np.ndarray,
        length: np.ndarray
) -> np.ndarray:
    line_segment = np.array(
        [
            [0, 0, 0],
            normal
        ]
    )
    line_segment = (line_segment * length) + centroid

    return line_segment


def select_rotator(click_position: np.ndarray, plane_normal: np.ndarray,  rotator_data: np.ndarray):
    # project the in view points onto the plane
    projected_points, projection_distances = project_points_onto_plane(
        points=rotator_data,
        plane_point=click_position,
        plane_normal=plane_normal,
    )

    # rotate points and plane to be axis aligned with normal [0, 0, 1]
    rotated_points, rotation_matrix = rotate_points(
        points=projected_points,
        current_plane_normal=plane_normal,
        new_plane_normal=[0, 0, 1],
    )
    rotated_click_point = np.dot(rotation_matrix, click_position)

    sizes = 2 * np.ones((rotator_data.shape[0],))
    # find the points the click intersects
    distances = abs(rotated_points[:, :2] - rotated_click_point[:2])
    in_slice_matches = np.all(
        distances <= np.expand_dims(sizes, axis=1) / 2,
        axis=1,
    )
    indices = np.where(in_slice_matches)[0]

    if len(indices) > 0:
        # find the point that is most in the foreground
        candidate_point_distances = projection_distances[indices]
        closest_index = indices[np.argmin(candidate_point_distances)]
        selection = closest_index
    else:
        selection = None
    return selection


def _frenet_frames(points, closed):
    """Calculates and returns the tangents, normals and binormals for
    the tube.

    From vispy: https://github.com/vispy/vispy/blob/main/vispy/visuals/tube.py
    """
    tangents = np.zeros((len(points), 3))
    normals = np.zeros((len(points), 3))

    epsilon = 0.0001

    # Compute tangent vectors for each segment
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    if not closed:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
    mags = np.sqrt(np.sum(tangents * tangents, axis=1))
    tangents /= mags[:, np.newaxis]

    # Get initial normal and binormal
    t = np.abs(tangents[0])

    smallest = np.argmin(t)
    normal = np.zeros(3)
    normal[smallest] = 1.

    vec = np.cross(tangents[0], normal)

    normals[0] = np.cross(tangents[0], vec)

    # Compute normal and binormal vectors along the path
    for i in range(1, len(points)):
        normals[i] = normals[i-1]

        vec = np.cross(tangents[i-1], tangents[i])
        if np.linalg.norm(vec) > epsilon:
            vec /= np.linalg.norm(vec)
            theta = np.arccos(np.clip(tangents[i-1].dot(tangents[i]), -1, 1))
            normals[i] = rotate(-np.degrees(theta),
                                vec)[:3, :3].dot(normals[i])

    if closed:
        theta = np.arccos(np.clip(normals[0].dot(normals[-1]), -1, 1))
        theta /= len(points) - 1

        if tangents[0].dot(np.cross(normals[0], normals[-1])) > 0:
            theta *= -1.

        for i in range(1, len(points)):
            normals[i] = rotate(-np.degrees(theta*i),
                                tangents[i])[:3, :3].dot(normals[i])

    binormals = np.cross(tangents, normals)

    return tangents, normals, binormals


def make_tube_mesh(
        points: np.ndarray,
        color: np.ndarray,
        radius: float = 1,
        closed:bool = False,
        tube_points: int=8,

) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make a mesh of a tube from a specified set of points.

    Modified from vispy:
    https://github.com/vispy/vispy/blob/main/vispy/visuals/tube.py

    Parameters
    ----------
    points
    tube_points

    Returns
    -------

    """
    # make sure we are working with floats
    points = np.asarray(points).astype(float)

    tangents, normals, binormals = _frenet_frames(points, closed)

    segments = len(points) - 1

    radius = [radius] * len(points)


    # get the positions of each vertex
    grid = np.zeros((len(points), tube_points, 3))
    for i in range(len(points)):
        pos = points[i]
        normal = normals[i]
        binormal = binormals[i]
        r = radius[i]

        # Add a vertex for each point on the circle
        v = np.arange(tube_points,
                      dtype=np.float) / tube_points * 2 * np.pi
        cx = -1. * r * np.cos(v)
        cy = r * np.sin(v)
        grid[i] = (pos + cx[:, np.newaxis] * normal +
                   cy[:, np.newaxis] * binormal)

    # construct the mesh
    indices = []
    for i in range(segments):
        for j in range(tube_points):
            ip = (i + 1) % segments if closed else i + 1
            jp = (j + 1) % tube_points

            index_a = i * tube_points + j
            index_b = ip * tube_points + j
            index_c = ip * tube_points + jp
            index_d = i * tube_points + jp

            indices.append([index_a, index_b, index_d])
            indices.append([index_b, index_c, index_d])

    vertices = grid.reshape(grid.shape[0] * grid.shape[1], 3)

    vertex_colors = np.repeat([color], len(vertices), axis=0)

    indices = np.array(indices, dtype=np.uint32)

    return vertices, indices, vertex_colors


def make_rotator_meshes(
    centroids: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    rotator_radius: float,
    tube_radius: float,
    tube_points: int,
    n_segments: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = []
    indices = []
    vertex_colors = []
    triangle_rotator_indices = []
    # we need to shift indices since we are concatentating multiple meshes
    curr_base_index = 0
    for i, (centroid, normal, color) in enumerate(zip(centroids, normals, colors)):
        points = create_circle_line_segments(
            centroid=centroid,
            normal=normal,
            radius=rotator_radius,
            n_segments=n_segments
        )
        rotator_vert, rotator_ind, rotator_colors = make_tube_mesh(
            points=points,
            radius=tube_radius,
            closed=True,
            color=color,
            tube_points=tube_points,
        )
        vertices.append(rotator_vert)
        indices.append(rotator_ind + curr_base_index)
        triangle_rotator_indices.append([i] * len(rotator_ind))
        vertex_colors.append(rotator_colors)
        curr_base_index += len(rotator_vert)

    return np.vstack(vertices), np.vstack(indices), np.vstack(vertex_colors), np.concatenate(triangle_rotator_indices)


def make_translator_meshes(
        centroids: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray,
        translator_length: float,
        tube_radius: float,
        tube_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = []
    indices = []
    vertex_colors = []
    triangle_translator_indices = []
    curr_base_index = 0

    for i, (centroid, normal, color) in enumerate(zip(centroids, normals, colors)):
        points = create_axis_line_segment(
            centroid=centroid,
            normal=normal,
            length=translator_length
        )
        translator_vert, translator_ind, translator_colors = make_tube_mesh(
            points=points,
            radius=tube_radius,
            closed=False,
            color=color,
            tube_points=tube_points,
        )
        vertices.append(translator_vert)
        indices.append(translator_ind + curr_base_index)
        vertex_colors.append(translator_colors)
        triangle_translator_indices.append([i] * len(translator_ind))
        curr_base_index += len(translator_vert)

    return np.vstack(vertices), np.vstack(indices), np.vstack(vertex_colors), np.concatenate(triangle_translator_indices)


def make_rotator_arc(
        center_point: np.ndarray,
        normal_vector: np.ndarray,
        radius: float,
        n_segments: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """Make the line data for a rotator arc.

    Parameters
    ----------
    center_point : np.ndarray
        (3,) array with the center point of the arc.
    normal_vector : np.ndarray
        (3,) array with the normal vector of the arc.
    radius : float
        The radius of the arc in data units.
    n_segments : int
        The number of segments to discretize the arc into.

    Returns
    -------
    arc_vertices : np.ndarray
        (n_segements, 3) array of the arc vertices
    arc_connections : np.ndarray
        (n_segments - 1, 2) array of the paairs connections between vertices.
        For example, [0, 1] connects vertices 0 and 1.
    """
    # create vertices with normal [1, 0, 0] centered at [0, 0, 0]
    t = np.linspace(0, np.pi / 2, n_segments)
    vertices = np.stack([0 * t, radius * np.sin(t), radius * np.cos(t)], 1).astype(np.float32)

    # transform the vertices
    rotation_matrix = rotation_matrix_from_vectors_3d(
        normal_vector,
        np.array([1, 0, 0])
    )
    rotated_vertices = vertices @ rotation_matrix.T
    arc_vertices = rotated_vertices + center_point

    # create the arc connections
    connections_start = np.arange(n_segments - 1)
    connections_end = connections_start + 1
    arc_connections = np.column_stack([connections_start, connections_end])

    return arc_vertices, arc_connections


def make_rotator_data(
        rotator_vectors: np.ndarray,
        rotator_colors: np.ndarray,
        center_point: np.ndarray,
        radius: float,
        n_segments: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the data for the vispy Line visual for the rotators.

    Parameters
    ----------
    rotator_vectors : np.ndarray
        (n, 3) array for the n rotators to be created.
    rotator_colors : np.ndarray
        (n, 4) array of RGBA colors for the n rotators to be created.
    center_point : np.ndarray
        (3,) array with the center point of the arc.
    radius : float
        The radius of the arc in data units.
    n_segments : int
        The number of segments to discretize the arc into.

    Returns
    -------
    rotator_vertices : np.ndarray
        (n_rotators * n_segments, 3) array containing the coordinates
        of all vertices.
    rotator_connections : np.ndarray
        (n_rotators * [n_segments - 1], 2) array containing the
        connections between arc vertices.
    rotator_colors : np.ndarray
        (n_rotators * n_segments, 4) array containing RGBA colors
        for all vertices.
    handle_points : np.ndarray
        (n_rotators, 3) array containing the coordinates of the handle for
        each rotator.
    handle_colors : np.ndarray
        (n_rotators, 4) RGBA array containing the color for each rotator handle.
    rotator_indices : np.ndarray
        The rotator index for each vertex.
    """
    # the handle is at the midpoint of the arc
    handle_index = int(n_segments / 2)

    vertex_offset = 0
    rotator_vertices = []
    rotator_connections = []
    colors = []
    handle_points = []
    handle_colors = []
    rotator_indicies = []
    for rotator_index, (normal_vector, color) in enumerate(zip(rotator_vectors, rotator_colors)):
        # get the vertices and connections
        vertices, connections = make_rotator_arc(
            center_point=center_point,
            normal_vector=normal_vector,
            radius=radius,
            n_segments=n_segments

        )
        rotator_vertices.append(vertices)
        rotator_connections.append(connections + vertex_offset)

        # make the colors
        assert color.shape == (4,)
        colors.append(np.tile(color, (n_segments, 1)))

        # get the handle point and color
        handle_points.append(vertices[handle_index])
        handle_colors.append(color)

        # add the rotator indices
        rotator_indicies.append([rotator_index] * n_segments)

        vertex_offset += n_segments

    return np.concatenate(rotator_vertices), np.concatenate(rotator_connections), np.concatenate(colors), np.stack(handle_points), np.stack(handle_colors), np.concatenate(rotator_indicies)


class RotatorDragManager:
    def __init__(self, normal_vector: np.ndarray, update_period: float = 0.03):

        # store the normal vector of the rotator
        self.normal_vector = normal_vector

        # the minimum time that must pass between updates in seconds
        self._update_period = update_period

    def start_drag(self, manipulator_centroid_coordinate: np.ndarray) -> None:
        pass

    def update_drag(self, click_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass
