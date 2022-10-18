from typing import Tuple, Optional

from napari.utils.geometry import inside_triangles, project_points_onto_plane, rotate_points, \
    rotation_matrix_from_vectors_3d
import numpy as np


def select_line_segment(
        line_segment_points: np.ndarray,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
        distance_threshold: float = 10
) -> np.ndarray:
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
    n_axes = int(len(rotated_points_2d) / 2)
    for i in range(n_axes):
        p_0 = rotated_points_2d[i * 2]
        p_1 = rotated_points_2d[i * 2 + 1]
        dist = distance_between_point_and_line_segment_2d(rotated_click_point_2d, p_0, p_1)
        distances.append(dist)
    distances = np.asarray(distances)
    # determine if any of the axes were clicked based on their width
    potential_matches = np.argwhere(distances < distance_threshold)

    return potential_matches


def distance_between_point_and_line_segment_2d(p, p1, p2):
    """Calculate the distance between a point p and a line segment p1, p2
    """
    x0 = p[0]
    y0 = p[1]
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    numerator = np.linalg.norm((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    denominator = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return numerator / denominator


def select_triangle_from_click(
        click_point: np.ndarray, view_direction: np.ndarray, triangles: np.ndarray
):
    """Determine if a line goes through any of a set of triangles.

    For example, this could be used to determine if a click was
    in a triangle of a mesh.

    Parameters
    ----------
    click_point : np.ndarray
        (3,) array containing the location that was clicked. This
        should be in the same coordinate system as the vertices.
    view_direction : np.ndarray
        (3,) array describing the direction camera is pointing in
        the scene. This should be in the same coordinate system as
        the vertices.
    triangles : np.ndarray
        (n, 3, 3) array containing the coordinates for the 3 corners
        of n triangles.

    Returns
    -------
    in_triangles : np.ndarray
        (n,) boolean array that is True of the ray intersects the triangle
    """
    vertices = triangles.reshape((-1, triangles.shape[2]))
    # project the vertices of the bound region on to the view plane
    vertices_plane, signed_distance_to_plane = project_points_onto_plane(
        points=vertices, plane_point=click_point, plane_normal=view_direction
    )

    # rotate the plane to make the triangles 2D
    rotation_matrix = rotation_matrix_from_vectors_3d(view_direction, [0, 0, 1])
    rotated_vertices = vertices_plane @ rotation_matrix.T

    rotated_vertices_2d = rotated_vertices[:, :2]
    rotated_triangles_2d = rotated_vertices_2d.reshape(-1, 3, 2)
    line_pos_2D = rotation_matrix.dot(click_point)[:2]

    candidate_matches = inside_triangles(rotated_triangles_2d - line_pos_2D)

    candidate_match_indices = np.argwhere(candidate_matches)

    n_matches = len(candidate_match_indices)
    if n_matches == 0:
        triangle_index = None
    elif n_matches == 1:
        triangle_index = candidate_match_indices[0]
    else:
        potential_match_distances = signed_distance_to_plane[candidate_match_indices]
        triangle_index = candidate_match_indices[np.argmin(potential_match_distances)]

    return triangle_index


def select_mesh_from_click(
        click_point: np.ndarray, view_direction: np.ndarray, triangles: np.ndarray,
        triangle_indices: np.ndarray
):
    selected_triangle = select_triangle_from_click(
        click_point=click_point,
        view_direction=view_direction,
        triangles=triangles
    )
    if selected_triangle is not None:
        selected_mesh = np.squeeze(triangle_indices[selected_triangle])
    else:
        selected_mesh = None

    return selected_mesh


def select_sphere_from_click(
    click_point: np.ndarray, view_direction: np.ndarray, sphere_centroids: np.ndarray, sphere_diameter: float
) -> Optional[int]:
    """Determine which, if any spheres are intersected by a click ray.

    If multiple spheres are intersected, the closest sphere to the click point
    (ray start) will be returned.

    Parameters
    ----------
    click_point : np.ndarray
        The point where the click ray originates.
    view_direction : np.ndarray
        The unit vector pointing in the direction the viewer is looking.
    sphere_centroids : np.ndarray
        The (n, 3) array of center points for the n points.
    sphere_diameter : float
        The diameter of all spheres. Must the same diameter for all spheres.

    Returns
    -------
    selection : Optional[int]
        The index for the sphere that was intersected.
        Returns None if no spheres are intersected.
    """
    # project the in view points onto the camera plane
    projected_points, projection_distances = project_points_onto_plane(
        points=sphere_centroids,
        plane_point=click_point,
        plane_normal=view_direction,
    )

    # rotate points and plane to be axis aligned with normal [0, 0, 1]
    rotated_points, rotation_matrix = rotate_points(
        points=projected_points,
        current_plane_normal=view_direction,
        new_plane_normal=[0, 0, 1],
    )
    rotated_click_point = np.dot(rotation_matrix, click_point)

    # find the points the click intersects
    n_spheres = len(sphere_centroids)
    handle_sizes = np.tile(sphere_diameter, (n_spheres, 1))
    distances = abs(rotated_points[:, :2] - rotated_click_point[:2])

    # the -1 accounts for the edge width
    in_slice_matches = np.all(
        distances <= (handle_sizes - 1 / 2) - 1.5,
        axis=1,
    )
    indices = np.where(in_slice_matches)[0]

    if len(indices) > 0:
        # find the point that is most in the foreground
        candidate_point_distances = projection_distances[indices]
        min_distance_index = np.argmin(candidate_point_distances)
        selection = indices[min_distance_index]
    else:
        selection = None

    return selection
