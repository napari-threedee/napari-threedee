from typing import Optional

import napari
from napari.layers.image._image_constants import VolumeDepiction
from napari.layers.labels._labels_mouse_bindings import draw as napari_draw
from napari.layers.labels._labels_utils import interpolate_coordinates, first_nonzero_coordinate
from napari.layers.labels._labels_constants import Mode
import numpy as np

from napari_threedee._backend.threedee_model import N3dComponent
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, \
    remove_mouse_callback_safe


class PlaneLabeler(N3dComponent):
    def __init__(
        self,
        viewer: napari.Viewer,
        image_layer: Optional[napari.layers.Image] = None,
        labels_layer: Optional[napari.layers.Labels] = None,
        enabled: bool = False
    ):
        self.viewer = viewer
        self.image_layer = image_layer
        if labels_layer is None:
            labels_layer = self._create_labels_layer()
        self.labels_layer = labels_layer
        self.enabled = enabled

    def _create_labels_layer(self) -> napari.layers.Labels:
        if self.image_layer is None:
            self.labels_layer = None
        else:
            return self.viewer.add_labels(np.zeros_like(self.image_layer.data, dtype=int))

    def set_layers(
        self,
        image_layer: napari.layers.Image,
        labels_layer: napari.layers.Labels
    ):
        self.image_layer = image_layer
        self.labels_layer = labels_layer

    def _on_mode_change(self, event):
        if event.mode in ("paint", "erase"):
            # replace the mouse callback with the one aware of the image layer rendering plane
            remove_mouse_callback_safe(self.labels_layer.mouse_drag_callbacks, napari_draw)
            add_mouse_callback_safe(self.labels_layer.mouse_drag_callbacks, self.draw)
        elif event.mode == "fill":
            remove_mouse_callback_safe(self.labels_layer.mouse_drag_callbacks, self.draw)
            add_mouse_callback_safe(self.labels_layer.mouse_drag_callbacks, napari_draw)

    def _on_enable(self):
        if self.labels_layer is None:
            return
        self.labels_layer.events.mode.connect(self._on_mode_change)

        # update the event callbacks
        if self.labels_layer.mode in ("paint", "erase"):
            # replace the mouse callback with the one aware of the image layer rendering plane
            remove_mouse_callback_safe(self.labels_layer.mouse_drag_callbacks, napari_draw)
            add_mouse_callback_safe(self.labels_layer.mouse_drag_callbacks, self.draw)

        # currently the plane painting must be done with n_edit_dims = 3
        # todo modify painting so it can be done 2D in plane
        self.labels_layer.n_edit_dims = 3

    def _on_disable(self):
        if self.labels_layer is None:
            return
        self.labels_layer.events.mode.disconnect(self._on_mode_change)

        # switch the mouse callbacks
        remove_mouse_callback_safe(self.labels_layer.mouse_drag_callbacks, self.draw)
        if self.labels_layer.mode in ("paint", "erase"):
            add_mouse_callback_safe(self.labels_layer.mouse_drag_callbacks, napari_draw)

    def _mouse_event_to_nd_line_plane_intersection(self, layer, event):
        painting_plane = self.image_layer.plane
        intersection_3d = painting_plane.intersect_with_line(
            line_position=layer._world_to_displayed_data(
                position=event.position, dims_displayed=event.dims_displayed
            ),
            line_direction=layer._world_to_displayed_data_ray(
                event.view_direction, event.dims_displayed
            ),
        )
        intersection_nd = np.copy(event.position)
        intersection_nd[event.dims_displayed] = intersection_3d
        return intersection_nd

    def _mouse_event_to_labels_coordinate(self, layer, event):
        """Return the data coordinate of a Labels layer mouse event in 2D or 3D.

        In 2D, this is just the event's position transformed by the layer's
        world_to_data transform.

        In 3D, a ray is cast in data coordinates, and the coordinate of the first
        nonzero value along that ray is returned. If the ray only contains zeros,
        None is returned.

        Parameters
        ----------
        layer : napari.layers.Labels
            The Labels layer.
        event : vispy MouseEvent
            The mouse event, containing position and view direction attributes.

        Returns
        -------
        coordinates : array of int or None
            The data coordinates for the mouse event.
        """
        ndim = self.viewer.dims.ndisplay
        if ndim == 2:
            coordinates = layer.world_to_data(event.position)
        elif ndim == 3 and self.image_layer._depiction == VolumeDepiction.PLANE:
            # image layer is in plane rendering, so we should paint on the plane
            return self._mouse_event_to_nd_line_plane_intersection(layer, event)
        else:  # 3d
            start, end = layer.get_ray_intersections(
                position=event.position,
                view_direction=event.view_direction,
                dims_displayed=layer._dims_displayed,
                world=True,
            )
            if start is None and end is None:
                return None
            coordinates = first_nonzero_coordinate(layer.data, start, end)
        return coordinates

    def draw(self, layer, event):
        """Draw with the currently selected label to a coordinate.

        This method have different behavior when draw is called
        with different labeling layer mode.

        In PAINT mode the cursor functions like a paint brush changing any
        pixels it brushes over to the current label. If the background label
        `0` is selected than any pixels will be changed to background and this
        tool functions like an eraser. The size and shape of the cursor can be
        adjusted in the properties widget.

        In FILL mode the cursor functions like a fill bucket replacing pixels
        of the label clicked on with the current label. It can either replace
        all pixels of that label or just those that are contiguous with the
        clicked on pixel. If the background label `0` is selected than any
        pixels will be changed to background and this tool functions like an
        eraser
        """
        ndisplay = self.viewer.dims.ndisplay
        coordinates = self._mouse_event_to_labels_coordinate(layer, event)
        if np.any(coordinates < 0) or np.any(coordinates >= layer.data.shape):
            # catch case when plane is not intersected
            return

        # on press
        if layer._mode == Mode.ERASE:
            new_label = layer._background_label
        else:
            new_label = layer.selected_label

        if coordinates is not None:
            if layer._mode in [Mode.PAINT, Mode.ERASE]:
                layer.paint(coordinates, new_label)
            elif layer._mode == Mode.FILL:
                layer.fill(coordinates, new_label)
        else:  # still add an item to undo queue
            # when dragging, if we start a drag outside the layer, we will
            # incorrectly append to the previous history item. We create a
            # dummy history item to prevent this.
            dummy_indices = (np.zeros(shape=0, dtype=int),) * layer.data.ndim
            layer._undo_history.append([(dummy_indices, [], [])])

        last_cursor_coord = coordinates
        yield

        layer._block_saving = True
        # on move
        while event.type == 'mouse_move':
            coordinates = self._mouse_event_to_labels_coordinate(layer, event)
            if np.any(coordinates < 0) or np.any(coordinates >= layer.data.shape):
                # catch case when plane is not intersected
                break

            if coordinates is not None or last_cursor_coord is not None:
                interp_coord = interpolate_coordinates(
                    last_cursor_coord, coordinates, layer.brush_size
                )
                for c in interp_coord:
                    if (
                            ndisplay == 3
                            and layer.data[tuple(np.round(c).astype(int))] == 0
                    ):
                        continue
                    if layer._mode in [Mode.PAINT, Mode.ERASE]:
                        layer.paint(c, new_label, refresh=False)
                    elif layer._mode == Mode.FILL:
                        layer.fill(c, new_label, refresh=False)
                layer.refresh()
            last_cursor_coord = coordinates
            yield

        # on release
        layer._block_saving = False
        undo_item = layer._undo_history[-1]
        if len(undo_item) == 1 and len(undo_item[0][0][0]) == 0:
            layer._undo_history.pop()