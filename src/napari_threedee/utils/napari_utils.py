from __future__ import annotations

import inspect
from functools import partial
from typing import Optional, Tuple

import magicgui
import napari
import napari.layers
import napari.viewer
import numpy as np
from magicgui.widgets import FunctionGui
from napari.layers import Points, Image
from napari.layers.utils.layer_utils import dims_displayed_world_to_layer

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


def point_in_layer_bounding_box(point, layer):
    """Determine whether an nD point is inside a layers nD bounding box.

    Parameters
    ----------
    point : np.ndarray
        (n,) array containing nD point coordinates to check.
    layer : napari.layers.Layer
        napari layer to get the bounding box from
    
    Returns
    -------
    bool
        True if the point is in the bounding box of the layer,
        otherwise False
    
    Notes
    -----
    For a more general point-in-bbox function, see:
        `napari_threedee.utils.geometry.point_in_bounding_box`
    """
    dims_displayed = get_dims_displayed(layer)
    bbox = layer._display_bounding_box(dims_displayed).T
    if np.any(point < bbox[0]) or np.any(point > bbox[1]):
        return False
    else:
        return True

def clamp_point_to_layer_bounding_box(point: np.ndarray, layer):
    """Ensure that a point is inside of the bounding box of a given layer. 
    If the point has a coordinate outside of the bounding box, the value 
    is clipped to the max extent of the bounding box.

    Parameters
    ----------
    point : np.ndarray
        n-dimensional point as an (n,) ndarray. Multiple points can
        be passed as an (n, D) array.
    layer : napari.layers.Layer
        napari layer to get the bounding box from

    Returns
    -------
    clamped_point : np.ndarray
        `point` clamped to the limits of the layer bounding box

    Notes
    -----
    This function is derived from the napari function:
        `napari.utils.geometry.clamp_point_to_bounding_box`
    """
    dims_displayed = get_dims_displayed(layer)
    bbox = layer._display_bounding_box(dims_displayed)
    clamped_point = np.clip(point, bbox[:, 0], bbox[:, 1] - 1)
    return clamped_point


def add_point_on_plane(
    viewer: napari.viewer.Viewer,
    points_layer: napari.layers.Points = None,
    image_layer: napari.layers.Image = None,
    replace_selected: bool = False,
):
    # Early exit if image_layer isn't visible
    if image_layer.visible is False or image_layer.depiction != 'plane':
        return

    # event.position and event.view_direction are in world (scaled) coordinates
    position_world = viewer.cursor.position
    view_direction_world = viewer.camera.view_direction
    ndim_world = len(position_world)
    dims_displayed_world = np.asarray(viewer.dims.displayed)[list(viewer.dims.displayed_order)]

    # use image layer data (pixel) coordinates, because that's what plane.position uses
    position_image_data_coord = np.asarray(image_layer.world_to_data(position_world))
    view_direction_image_data_coord = np.asarray(image_layer._world_to_data_ray(view_direction_world))

    dims_displayed_image_layer = np.asarray(dims_displayed_world_to_layer(
        dims_displayed_world,
        ndim_world=ndim_world,
        ndim_layer=image_layer.ndim,
    ))

    # Calculate 3d intersection of click with plane through data in image data (pixel) coordinates
    position_image_data_3d = position_image_data_coord[dims_displayed_image_layer]
    view_direction_image_data_3d = view_direction_image_data_coord[dims_displayed_image_layer]
    intersection_image_data_3d = image_layer.plane.intersect_with_line(
        line_position=position_image_data_3d,
        line_direction=view_direction_image_data_3d
    )

    # Check if click was on plane by checking if intersection occurs within image layer
    # data bounding box. If not, exit early.
    if not point_in_layer_bounding_box(intersection_image_data_3d, image_layer):
        return

    # Transform the intersection coordinate from image layer coordinates to world coordinates
    intersection_3d_world = np.asarray(image_layer.data_to_world(intersection_image_data_3d))[
        dims_displayed_image_layer]

    # convert to nd in world coordinates
    intersection_nd_world = np.asarray(viewer.dims.point)
    intersection_nd_world[dims_displayed_image_layer] = intersection_3d_world

    # Transform the the point in world coordinates to point layer data coordinates
    intersection_3d_points = points_layer.world_to_data(intersection_3d_world)
    intersection_nd_points = points_layer.world_to_data(intersection_nd_world)

    if replace_selected:
        points_layer.remove_selected()
    if points_layer.data.shape[-1] < len(intersection_nd_points):
        intersection_nd_points = intersection_3d_points
    points_layer.add(intersection_nd_points)
