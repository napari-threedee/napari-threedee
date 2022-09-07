import numpy as np
import pytest

from napari_threedee.utils.geometry import signed_angle_between_vectors, rotation_matrix_around_vector_3d


@pytest.mark.parametrize(
    "vector_0,vector_1,rotation_axis,expected_angle",
    [
        (np.array([0, 1, 1]), np.array([0, -1, 1]), np.array([1, 0, 0]), np.pi/2),
        (np.array([0, 1, 1]), np.array([0, -1, 1]), np.array([-1, 0, 0]), -np.pi/2),
        (np.array([0, 1, 1]), np.array([0, 1, -1]), np.array([1, 0, 0]), -np.pi/2),
        (np.array([0, 1, 1]), np.array([0, 1, -1]), np.array([-1, 0, 0]), np.pi/2)
    ]
)
def test_signed_angle_between_vectors(vector_0, vector_1, rotation_axis, expected_angle):
    angle = signed_angle_between_vectors(
        vector_0=vector_0,
        vector_1=vector_1,
        rotation_axis=rotation_axis
    )

    np.testing.assert_allclose(angle, expected_angle)


def test_rotation_matrix_around_vector_3d():
    angle = np.pi / 2
    rotation_matrix = rotation_matrix_around_vector_3d(angle=angle, vector=np.array([1, 0, 0]))

    expected_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ]
    )
    np.testing.assert_allclose(rotation_matrix, expected_matrix)
