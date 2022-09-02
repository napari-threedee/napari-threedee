from __future__ import annotations

from typing import TYPE_CHECKING
from napari.layers.utils.interactivity_utils import drag_data_to_projected_distance
import numpy as np

from .geometry_utils import point_in_bounding_box

if TYPE_CHECKING:
    import napari
    import napari.layers
    from napari.utils.events import Event


def add_point_on_plane(
        viewer: napari.viewer.Viewer,
        event: Event,
        points_layer: napari.layers.Points = None,
        plane_layer: napari.layers.Image = None,
        append: bool = True,
):
    # Early exit if not alt-clicked
    if 'Alt' not in event.modifiers:
        return

    # Early exit if plane_layer isn't visible
    if not plane_layer.visible:
        return

    # Calculate intersection of click with plane through data in displayed data (scene) coordinates
    displayed_dims = np.asarray(viewer.dims.displayed)[list(viewer.dims.displayed_order)]
    cursor_position_3d = np.asarray(event.position)[displayed_dims]
    intersection_3d = plane_layer.plane.intersect_with_line(
        line_position=cursor_position_3d,
        line_direction=event.view_direction[displayed_dims]
    )
    intersection_nd = np.asarray(viewer.dims.point)
    intersection_nd[displayed_dims] = intersection_3d

    # Check if click was on plane by checking if intersection occurs within
    # data bounding box. If not, exit early.
    if not point_in_bounding_box(intersection_nd, plane_layer.extent.data):
        return

    if append:
        points_layer.add(intersection_nd)
    else:
        points_layer.data = intersection_nd
