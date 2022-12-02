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


def get_vispy_node(viewer, layer):
    """"""
    napari_visual = get_napari_visual(viewer, layer)

    if isinstance(layer, Image):
        return napari_visual._layer_node.get_node(3)
    elif isinstance(layer, Points):
        return napari_visual.node


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


