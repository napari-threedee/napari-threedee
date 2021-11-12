import numpy as np


def point_in_bounding_box(point: np.ndarray, bounding_box: np.ndarray) -> bool:
    """Determine whether an nD point is inside an nD bounding box.
    Parameters
    ----------
    point : np.ndarray
        (n,) array containing nD point coordinates to check.
    bounding_box : np.ndarray
        (2, n) array containing the min and max of the nD bounding box.
        As returned by `Layer._extent_data`.
    """
    if np.all(point > bounding_box[0]) and np.all(point < bounding_box[1]):
        return True
    return False


def point_in_layer_bounding_box(point, layer):
    bbox = layer._display_bounding_box(layer._dims_displayed).T
    if np.any(point < bbox[0]) or np.any(point > bbox[1]):
        return False
    else:
        return True
