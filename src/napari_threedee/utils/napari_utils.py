from typing import Optional

from napari.layers import Points, Image


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
