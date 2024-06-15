from __future__ import annotations

from typing import TYPE_CHECKING
from napari.layers.utils.interactivity_utils import drag_data_to_projected_distance

from napari_threedee.utils.napari_utils import add_point_on_plane

if TYPE_CHECKING:
    import napari
    import napari.layers
    from napari.utils.events import Event


def on_mouse_alt_click_add_point_on_plane(
        viewer: napari.viewer.Viewer,
        event: Event,
        points_layer: napari.layers.Points = None,
        image_layer: napari.layers.Image = None,
        replace_selected: bool = False,
):
    # Early exit if not alt-clicked
    if 'Alt' not in event.modifiers:
        return

    add_point_on_plane(
        viewer=viewer,
        points_layer=points_layer,
        image_layer=image_layer,
        replace_selected=replace_selected
    )

