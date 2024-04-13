import inspect
from functools import partial
from typing import Optional, Tuple

import magicgui
import napari
import numpy as np
from magicgui.widgets import FunctionGui
from napari.layers import Points, Image


NAPARI_LAYER_TYPES = (
    napari.layers.Layer,
    napari.layers.Points,
    napari.layers.Image,
    napari.layers.Surface,
    napari.layers.Shapes,
    napari.layers.Vectors,
    napari.layers.Tracks,
)


def get_napari_visual(viewer, layer):
    """Get the visual class for a given layer

    Parameters
    ----------
    viewer
        The napari viewer object
    layer
        The napari layer object for which to find the visual.

    Returns
    -------
    visual
        The napari visual class for the layer.
    """
    visual = viewer.window._qt_window._qt_viewer.layer_to_visual[layer]

    return visual


def get_vispy_layer_node(viewer: napari.Viewer, layer):
    """Get the vispy node associated with a layer"""
    napari_visual = get_napari_visual(viewer, layer)

    if isinstance(layer, Image):
        return napari_visual._layer_node.get_node(3)
    elif isinstance(layer, Points):
        return napari_visual.node


def get_vispy_root_node(viewer: napari.Viewer, layer):
    """Get the vispy node at the root of the scene graph.

    This is the node that layers are added to.
    """
    # this will need to be updated in napari 0.5.0
    # viewer.window._qt_window._qt_viewer.canvas.view.scene
    qt_viewer = viewer.window._qt_window._qt_viewer
    return qt_viewer.view.scene


def remove_mouse_callback_safe(callback_list, callback):
    if callback in callback_list:
        callback_list.remove(callback)


def add_mouse_callback_safe(callback_list, callback, index: Optional[int] = None):
    if callback not in callback_list:
        if index is not None:
            callback_list.insert(index, callback)
        else:
            callback_list.append(callback)


def get_layers_of_type(*args, viewer: napari.Viewer, layer_type):
    return [layer for layer in viewer.layers if isinstance(layer, layer_type)]


def get_dims_displayed(layer):
    # layer._dims_displayed was removed in
    # https://github.com/napari/napari/pull/5003
    if hasattr(layer, "_slice_input"):
        return layer._slice_input.displayed
    return layer._dims_displayed


def generate_populated_layer_selection_widget(func, viewer) -> FunctionGui:
    parameters = inspect.signature(func).parameters
    magicgui_parameter_arguments = {
        parameter_name: {
            'choices': partial(get_layers_of_type, viewer=viewer, layer_type=parameter.annotation)
        }
        for parameter_name, parameter
        in parameters.items()
        if parameter.annotation in NAPARI_LAYER_TYPES
    }
    return magicgui.magicgui(func, **magicgui_parameter_arguments, auto_call=True)


def get_mouse_position_in_displayed_dimensions(event) -> np.ndarray:
    """Get the position under the mouse in scene (displayed world) coordinates.

    Parameters
    ----------
    event
        The mouse event.

    Returns
    -------
    click_dir_data_3d : np.ndarray
        The click direction in displayed data coordiantes
    """
    click_position_world = event.position
    return np.asarray(click_position_world)[list(event.dims_displayed)]


def get_view_direction_in_displayed_dimensions(event) -> np.ndarray:
    """Get the view direction under the mouse in scene (displayed world) coordinates.

    Parameters
    ----------
    event: Event
        napari mouse event.
    """
    view_direction_world = event.view_direction
    return np.asarray(view_direction_world)[list(event.dims_displayed)]


def get_mouse_position_in_displayed_layer_data_coordinates(layer, event) -> Tuple[np.ndarray, np.ndarray]:
    """Get the mouse click position and direction in layer data displayed coordinates.

    Parameters
    ----------
    layer : napari.layers.Layer
        The layer to convert the coordinates to.
    event
        The mouse event.

    Returns
    -------
    click_position_data_3d : np.ndarray
        The click position in displayed data coordinates.
    click_dir_data_3d : np.ndarray
        The click direction in displayed data coordiantes
    """
    click_position_world = event.position
    click_position_data_3d = np.asarray(
        layer._world_to_displayed_data(
            click_position_world,
            event.dims_displayed
        )
    )
    click_dir_data_3d = np.asarray(
        layer._world_to_displayed_data_ray(
            event.view_direction,
            event.dims_displayed
        )
    )

    return click_position_data_3d, click_dir_data_3d


def data_to_world_ray(vector, layer):
    """Convert a vector defining an orientation from data coordinates to world coordinates.
    For example, this would be used to convert the view ray.

    Parameters
    ----------
    vector : tuple, list, 1D array
        A vector in data coordinates.
    layer : napari.layers.BaseLayer
        The napari layer to get the transform from.

    Returns
    -------
    np.ndarray
        Transformed vector in data coordinates.
    """
    p1 = np.asarray(layer.data_to_world(vector))
    p0 = np.asarray(layer.data_to_world(np.zeros_like(vector)))
    normalized_vector = (p1 - p0) / np.linalg.norm(p1 - p0)

    return normalized_vector


def data_to_world_normal(vector, layer):
    """Convert a normal vector defining an orientation from data coordinates to world coordinates.
    For example, this would be used to a plane normal.

    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals.html

    Parameters
    ----------
    vector : tuple, list, 1D array
        A vector in data coordinates.
    layer : napari.layers.BaseLayer
        The napari layer to get the transform from.

    Returns
    -------
    np.ndarray
        Transformed vector in data coordinates. This returns a unit vector.
    """
    unit_vector = np.asarray(vector) / np.linalg.norm(vector)

    # get the transform
    inverse_transform = layer._transforms[1:].simplified.inverse.linear_matrix
    transpose_inverse_transform = inverse_transform.T

    # transform the vector
    transformed_vector = np.matmul(transpose_inverse_transform, unit_vector)

    return transformed_vector / np.linalg.norm(transformed_vector)


def world_to_data_normal(vector, layer):
    """Convert a normal vector defining an orientation from world coordinates to data coordinates.
    For example, this would be used to a plane normal.

    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals.html

    Parameters
    ----------
    vector : tuple, list, 1D array
        A vector in world coordinates.
    layer : napari.layers.BaseLayer
        The napari layer to get the transform from.

    Returns
    -------
    np.ndarray
        Transformed vector in data coordinates. This returns a unit vector.
    """
    unit_vector = np.asarray(vector) / np.linalg.norm(vector)

    # get the transform
    # the napari transform is from layer -> world.
    # We want the inverse of the world ->  layer, so we just take the napari transform
    inverse_transform = layer._transforms[1:].simplified.linear_matrix
    transpose_inverse_transform = inverse_transform.T

    # transform the vector
    transformed_vector = np.matmul(transpose_inverse_transform, unit_vector)

    return transformed_vector / np.linalg.norm(transformed_vector)