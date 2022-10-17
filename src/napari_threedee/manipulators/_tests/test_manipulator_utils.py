import numpy as np

from napari_threedee.manipulators.manipulator_utils import make_rotator_arc, make_rotator_data


def test_make_rotator_arc():
    center_point = np.array([10, 20, 30])
    radius = 5
    normal_vector = np.array([0, 1, 0])
    n_segments = 70
    vertices, connections = make_rotator_arc(
        center_point=center_point,
        radius=radius,
        normal_vector=normal_vector,
        n_segments=n_segments

    )

    assert vertices.shape == (n_segments, 3)
    assert connections.shape == (n_segments - 1, 2)

    # ensure the arc has the correct radius
    radii = np.linalg.norm(vertices - center_point, axis=1)
    np.testing.assert_allclose(radii, radius)

    # ensure the arc is in the correct plane
    np.testing.assert_allclose(vertices[:, 1], center_point[1])


def test_make_rotator_data():
    rotator_normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rotator_colors = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1]
        ]
    )
    center_point = np.array([10, 10, 10])
    radius = 5
    n_segments = 50
    n_rotators = 3

    vertices, connections, colors, handle_points, handle_colors, rotator_indices = make_rotator_data(
        rotator_vectors=rotator_normals,
        rotator_colors=rotator_colors,
        center_point=center_point,
        radius=radius,
        n_segments=n_segments
    )

    # check that arrays were stacked/concatenated correctly
    n_expected_vertices = n_rotators * n_segments
    assert vertices.shape == (n_expected_vertices, 3)
    assert connections.shape == (n_rotators * (n_segments - 1), 2)
    assert colors.shape == (n_expected_vertices, 4)
    assert handle_points.shape == (n_rotators, 3)
    assert handle_colors.shape == (n_rotators, 4)

    # check that vertex offset was done correction
    assert connections[n_segments-1, 0] == n_segments
