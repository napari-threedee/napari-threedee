from __future__ import annotations

from typing import TYPE_CHECKING
from napari.layers.utils.interactivity_utils import drag_data_to_projected_distance
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
    # Calculate intersection of click with plane through data in displayed data (scene) coordinates
    displayed_dims = np.asarray(viewer.dims.displayed)[list(viewer.dims.displayed_order)]
    # use data coordinates, because that's what plane.position uses data (displayed) coords
    cursor_data_coord = image_layer.world_to_data(event.position)
    view_direction_data_coord = image_layer._world_to_data_ray(event.view_direction)
    cursor_position_3d = cursor_data_coord[displayed_dims]
    intersection_3d = image_layer.plane.intersect_with_line(
        line_position=cursor_position_3d,
        line_direction=view_direction_data_coord[displayed_dims]
    )
    intersection_nd = np.asarray(viewer.dims.point)
    intersection_nd[displayed_dims] = intersection_3d

    # Check if click was on plane by checking if intersection occurs within
    # data bounding box. If not, exit early.
    if not point_in_bounding_box(intersection_nd, image_layer.extent.data):
        return

    if replace_selected:
        points_layer.remove_selected()
    if points_layer.data.shape[-1] < len(intersection_nd):
        intersection_nd = intersection_3d
    points_layer.add(intersection_nd)
