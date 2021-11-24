import numpy as np
from napari.utils.geometry import project_points_onto_plane, rotate_points
from napari.utils.translations import trans


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
