from napari.utils.geometry import project_points_onto_plane, rotate_points
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

    numerator = np.linalg.norm((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))
    denominator = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    return numerator / denominator