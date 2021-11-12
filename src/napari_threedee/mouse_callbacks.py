from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from napari.layers.utils.interactivity_utils import drag_data_to_projected_distance
from napari.utils.geometry import clamp_point_to_bounding_box
import numpy as np

from .geometry_utils import point_in_bounding_box

if TYPE_CHECKING:
    import napari
    import napari.layers
    from napari.utils.events import Event


def shift_plane_along_normal(
        viewer: napari.Viewer,
        event: Event,
        layer: napari.layers.Image
):
    """Shift a rendered plane along its normal vector.
    This function will shift a plane along its normal vector when the plane is
    clicked and dragged."""
    # Calculate intersection of click with plane through data in data coordinates
    intersection = layer.experimental_slicing_plane.intersect_with_line(
        line_position=event.position,
        line_direction=event.view_direction
    )

    # Check if click was on plane by checking if intersection occurs within
    # data bounding box. If not, exit early.
    if not point_in_bounding_box(intersection, layer.extent.data):
        return

    # Store original plane position and disable interactivity during plane drag
    original_plane_position = np.copy(layer.experimental_slicing_plane.position)
    layer.interactive = False

    # Store mouse position at start of drag
    start_position = np.copy(event.position)
    yield

    while event.type == 'mouse_move':
        current_position = event.position
        current_view_direction = event.view_direction
        current_plane_normal = layer.experimental_slicing_plane.normal

        # Project mouse drag onto plane normal
        drag_distance = drag_data_to_projected_distance(
            start_position=start_position,
            end_position=current_position,
            view_direction=current_view_direction,
            vector=current_plane_normal,
        )

        # Calculate updated plane position
        updated_position = original_plane_position + (
                drag_distance * np.array(layer.experimental_slicing_plane.normal)
        )

        clamped_plane_position = clamp_point_to_bounding_box(
            updated_position, layer._display_bounding_box(event.dims_displayed)
        )

        layer.experimental_slicing_plane.position = clamped_plane_position
        yield

    # Re-enable volume_layer interactivity after the drag
    layer.interactive = True


def add_point_on_plane(
        viewer,
        event,
        points_layer: napari.layers.Points = None,
        plane_layer: napari.layers.Image = None,
        append: bool = True,
):
    # Early exit if not alt-clicked
    if 'Alt' not in event.modifiers:
        return

    # Early exit if volume_layer isn't visible
    if not plane_layer.visible:
        return

    # Ensure added points will be visible until plane depth is sorted
    points_layer.blending = 'translucent_no_depth'

    # Calculate intersection of click with plane through data in data coordinates
    intersection = plane_layer.experimental_slicing_plane.intersect_with_line(
        line_position=viewer.cursor.position, line_direction=viewer.cursor._view_direction
    )

    # Check if click was on plane by checking if intersection occurs within
    # data bounding box. If not, exit early.
    if not point_in_bounding_box(intersection, plane_layer.extent.data):
        return

    if append:
        points_layer.add(intersection)
    else:
        points_layer.data = intersection
        # points_layer.add(intersection)
