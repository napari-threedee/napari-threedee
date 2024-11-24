import napari.layers
from napari.utils.geometry import rotation_matrix_from_vectors_3d
from napari.layers.utils.plane import ClippingPlane
import numpy as np
from napari_threedee.manipulators.base_manipulator import BaseManipulator
from napari_threedee.utils.geometry import point_in_bounding_box
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, remove_mouse_callback_safe, data_to_world_normal, world_to_data_normal


class ClippingPlaneManipulator(BaseManipulator):
    """A manipulator for moving and orienting a layer clipping plane."""

    def __init__(self, viewer, layer, clipping_plane_idx = None):
        # if a clipping plane is not provided, append a new clipping plane and use that
        if clipping_plane_idx is None:
            layer.experimental_clipping_planes.append(ClippingPlane(enabled=True))
            self.clipping_plane = layer.experimental_clipping_planes[-1]
        # if a clipping plane index is provided, use that
        elif clipping_plane_idx <= len(layer.experimental_clipping_planes) -1:
            self.clipping_plane = layer.experimental_clipping_planes[clipping_plane_idx]
        else:
            raise ValueError("Clipping plane index out of bounds")

        super().__init__(viewer, layer, rotator_axes='xyz', translator_axes='z')


    def set_layers(self, layers: napari.layers.Layer):
        super().set_layers(layers)

    def _connect_events(self):
        self.clipping_plane.events.position.connect(self._update_transform)
        self.clipping_plane.events.normal.connect(self._update_transform)
        self.layer.events.visible.connect(self._on_visibility_change)
        self._viewer.layers.events.removed.connect(self._disable_and_remove)
        add_mouse_callback_safe(self.layer.mouse_double_click_callbacks, self._double_click_callback)

    def _disconnect_events(self):
        self.clipping_plane.events.position.disconnect(self._update_transform)
        self.clipping_plane.events.normal.disconnect(self._update_transform)
        remove_mouse_callback_safe(self.layer.mouse_double_click_callbacks, self._double_click_callback)

    def _double_click_callback(self, layer, event):
        """Set the plane (and manipulator position) on double click.
        Based on napari.layers.image._image_mouse_bindings.set_plane_position
        Can be removed once napari implements this for clipping planes
        """
         
        intersection = self.clipping_plane.intersect_with_line(
            line_position=np.asarray(event.position)[event.dims_displayed],
            line_direction=np.asarray(event.view_direction)[event.dims_displayed],
        )

        # Check if click was on plane and if not, exit early.
        if not point_in_bounding_box(
            intersection, self.layer.extent.data[:, event.dims_displayed]
        ):
            return

        self.clipping_plane.position = intersection

    def _update_transform(self):
        # get the new transformation data
        self._initialize_transform()

        # redraw
        self._backend._on_transformation_changed()

    def _initialize_transform(self):
        origin_world = self.layer.data_to_world(self.clipping_plane.position)
        self.origin = np.array(origin_world)
        plane_normal_data = self.clipping_plane.normal
        plane_normal_world = data_to_world_normal(vector=plane_normal_data, layer=self.layer)
        manipulator_normal = -1 * plane_normal_world
        self.rotation_matrix = rotation_matrix_from_vectors_3d([1, 0, 0], manipulator_normal)


    def _while_dragging_translator(self):
        with self.clipping_plane.events.position.blocker(self._update_transform):
            self.clipping_plane.position = self.layer.world_to_data(self.origin)

    def _while_dragging_rotator(self):
        with self.clipping_plane.events.normal.blocker(self._update_transform):
            z_vector_data = world_to_data_normal(vector=self.z_vector, layer=self.layer)
            self.clipping_plane.normal = -1 * z_vector_data
