from typing import Optional

import napari
import numpy as np
from napari.layers import Image, Points, Surface
from napari.utils.events import EmitterGroup, Event
from vispy.geometry import create_sphere

from napari_threedee._base_model import ThreeDeeModel
from napari_threedee.mouse_callbacks import add_point_on_plane
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, remove_mouse_callback_safe


class SphereAnnotator(ThreeDeeModel):
    COLOR_CYCLE = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
    ]
    # column name in the points layer features to store the sphere ID
    SPHERE_ID_COLUMN = "sphere_id"
    SPHERE_RADIUS_COLUMN = "radius"
    SPHERE_METADATA_LABEL = "sphere_data"
    DEFAULT_SPHERE_RADIUS = 5

    def __init__(
        self,
        viewer: napari.Viewer,
        image_layer: Optional[Image] = None,
        enabled: bool = False
    ):
        self.events = EmitterGroup(
            source=self,
            current_spline_id=Event
        )

        self.viewer = viewer
        self.image_layer = image_layer
        self.points_layer = None
        self.enabled = enabled

        self.current_sphere_id: int = 0

        if image_layer is not None:
            self.set_layers(self.image_layer)

    @property
    def current_sphere_id(self):
        return self._current_sphere_id

    @current_sphere_id.setter
    def current_sphere_id(self, id: int):
        self._current_sphere_id = np.clip(id, 0, None)
        if self.points_layer is not None:
            self.points_layer.selected_data = {}
            self.points_layer.current_properties = {
                self.SPHERE_ID_COLUMN: self.current_sphere_id,
                self.SPHERE_RADIUS_COLUMN: [self.DEFAULT_SPHERE_RADIUS]
            }
        self.events.current_spline_id()

    def next_sphere(self, event=None):
        self.current_sphere_id += 1

    def previous_sphere(self, event=None):
        self.current_sphere_id -= 1

    def _mouse_callback(self, viewer, event):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        add_point_on_plane(
            viewer=viewer,
            event=event,
            points_layer=self.points_layer,
            plane_layer=self.image_layer,
            replace_selected=True,
        )

    def _create_points_layer(self) -> Optional[Points]:
        layer = Points(
            data=[0] * self.image_layer.data.ndim,
            ndim=self.image_layer.data.ndim,
            name="sphere centers",
            size=7,
            features={
                self.SPHERE_ID_COLUMN: [0],
                self.SPHERE_RADIUS_COLUMN: [self.DEFAULT_SPHERE_RADIUS]},
            face_color=self.SPHERE_ID_COLUMN,
            face_color_cycle=self.COLOR_CYCLE,
            metadata={"splines": dict()}
        )
        layer.selected_data = {0}
        layer.remove_selected()
        self.current_sphere_id = self.current_sphere_id
        return layer

    def _create_surface_layer(self) -> Surface:
        return Surface(
            data=(np.array([[0, 0, 0]]), np.array([[0, 0, 0]])),
            name="spheres",
            opacity=0.7,
        )

    def set_layers(self, image_layer: napari.layers.Image):
        self.image_layer = image_layer
        if self.points_layer is None and self.image_layer is not None:
            self.points_layer = self._create_points_layer()
            self.viewer.add_layer(self.points_layer)
            self.surface_layer = self._create_surface_layer()
            self.viewer.add_layer(self.surface_layer)
            self.viewer.layers.selection.active = self.points_layer

    def _on_enable(self):
        if self.points_layer is not None:
            add_mouse_callback_safe(
                callback_list=self.viewer.mouse_drag_callbacks,
                callback=self._mouse_callback
            )
            self.points_layer.events.data.connect(self._on_point_data_changed)
            self.viewer.bind_key('n', self.next_sphere)
            self.viewer.layers.selection.active = self.image_layer

    def _on_disable(self):
        remove_mouse_callback_safe(
            callback_list=self.viewer.mouse_drag_callbacks,
            callback=self._mouse_callback
        )
        if self.points_layer is not None:
            self.points_layer.events.data.disconnect(self._on_point_data_changed)
        self.viewer.bind_key('n', None)

    def _on_point_data_changed(self, event=None):
        self._update_spheres()
        self._draw_spheres()

    def _update_spheres(self):
        grouped_points_features = self.points_layer.features.groupby(self.SPHERE_ID_COLUMN)
        sphere_vertices = []
        sphere_faces = []
        face_index_offset = 0
        for sphere_id, sphere_df in grouped_points_features:
            point_index = int(sphere_df.index.values[0])
            position = self.points_layer.data[point_index]
            radius = float(sphere_df[self.SPHERE_RADIUS_COLUMN].iloc[0])
            mesh_data = create_sphere(radius=radius)
            vertex_data = mesh_data.get_vertices() + position
            sphere_vertices.append(vertex_data)
            sphere_faces.append(mesh_data.get_faces() + face_index_offset)
            face_index_offset += len(vertex_data)
        if len(sphere_vertices) > 0:
            sphere_vertices = np.concatenate(sphere_vertices, axis=0)
            sphere_faces = np.concatenate(sphere_faces, axis=0)
            self.points_layer.metadata[self.SPHERE_METADATA_LABEL] = (
                sphere_vertices, sphere_faces
            )
        else:
            self.points_layer.metadata[self.SPHERE_METADATA_LABEL] = None

    def _draw_spheres(self):
        if self.points_layer.metadata[self.SPHERE_METADATA_LABEL] is not None:
            self.surface_layer.data = self.points_layer.metadata[self.SPHERE_METADATA_LABEL]
