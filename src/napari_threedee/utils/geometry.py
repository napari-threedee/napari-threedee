import numpy as np


def signed_angle_between_vectors(vector_0, vector_1, rotation_axis: np.ndarray) -> float:
    """ Returns the angle in radians between vectors 'v1' and 'v2'.

    Parameters
    ----------
    vector_0 : np.ndarray
        The vector to start the rotation at.
    vector_1 : np.ndarray
        The vector the rotation ends at.
    rotation_axis : np.ndarray
        The axis around which the rotation is occuring.
        Must be orthogonal to vector_0 and vector_1.

    Returns
    -------
    angle : float
        The signed angle of rotation in radians.
    """
    unit_vector_0 = vector_0 / np.linalg.norm(vector_0)
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)

    unsigned_angle = np.arccos(np.clip(np.dot(unit_vector_0, unit_vector_1), -1.0, 1.0))

    v3 = np.cross(unit_vector_0, unit_vector_1)

    angle_sign = -1 if np.dot(v3, rotation_axis) < 0 else 1

    return angle_sign * unsigned_angle


def rotation_matrix_around_vector_3d(angle: float, vector: np.ndarray) -> np.ndarray:
    """Create the rotation matrix for a specified angle of rotation around a vector.

    Parameters
    ----------
    angle : float
        The signed angle of rotation in radians.

    vector : np.ndarray
        The vector around which to perform the rotation.

    Returns
    -------
    rotation_matrix : np.ndarray
        (3, 3) rotation matrix for the specified rotation.
    """
    vector_u = vector / np.linalg.norm(vector)
    u_0 = vector_u[0]
    u_1 = vector_u[1]
    u_2 = vector_u[2]

    cos_term = 1 - np.cos(angle)
    sine_term = np.sin(angle)

    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, 0] = np.cos(angle) + (u_0**2) * cos_term
    rotation_matrix[0, 1] = (u_0 * u_1 * cos_term) - u_2 * sine_term
    rotation_matrix[0, 2] = (u_0 * u_2 * cos_term) + u_1 * sine_term

    rotation_matrix[1, 0] = (u_1 * u_0 * cos_term) + u_2 * sine_term
    rotation_matrix[1, 1] = np.cos(angle) + (u_1**2) * cos_term
    rotation_matrix[1, 2] = (u_1 * u_2 * cos_term) - u_0 * sine_term

    rotation_matrix[2, 0] = (u_0 * u_2 * cos_term) - u_1 * sine_term
    rotation_matrix[2, 1] = u_1 * u_2 * cos_term + u_0 * sine_term
    rotation_matrix[2, 2] = np.cos(angle) + (u_2**2) * cos_term

    return rotation_matrix
