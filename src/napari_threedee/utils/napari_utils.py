import inspect
from functools import partial
from typing import Optional

import magicgui
import napari
from magicgui.widgets import FunctionGui
from napari.layers import Points, Image
from qtpy.QtWidgets import QWidget

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
    mgui_param_args = {
        parameter_name: {
            'choices': partial(get_layers_of_type, viewer=viewer, layer_type=parameter.annotation)
        }
        for parameter_name, parameter
        in parameters.items()
        if parameter.annotation in NAPARI_LAYER_TYPES
    }
    return magicgui.magicgui(func, **mgui_param_args, auto_call=True)

