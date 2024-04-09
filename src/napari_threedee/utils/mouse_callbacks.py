from __future__ import annotations

from typing import TYPE_CHECKING
from napari.layers.utils.interactivity_utils import drag_data_to_projected_distance
from napari.layers.utils.layer_utils import dims_displayed_world_to_layer
import numpy as np

from napari_threedee.utils.geometry import point_in_bounding_box

if TYPE_CHECKING:
    import napari
    import napari.layers
    from napari.utils.events import Event


def add_point_on_plane(
        viewer: napari.viewer.Viewer,
        event: Event,
        points_layer: napari.layers.Points = None,
        image_layer: napari.layers.Image = None,
        replace_selected: bool = False,
):
    # Early exit if not alt-clicked
    if 'Alt' not in event.modifiers:
        return

    # Early exit if image_layer isn't visible
    if image_layer.visible is False or image_layer.depiction != 'plane':
        return
    # event.position and event.view_direction are in world (scaled) coordinates
    position_world = event.position
    view_direction_world = event.view_direction
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
    if not point_in_bounding_box(intersection_image_data_3d, image_layer.extent.data[:, dims_displayed_image_layer]):
        return

    # Transform the intersection coordinate from image layer coordinates to world coordinates
    intersection_3d_world = np.asarray(image_layer.data_to_world(intersection_image_data_3d))[dims_displayed_image_layer]

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
