import warnings
from enum import Enum, auto
from typing import Optional, Union

import napari
import numpy as np
from napari.layers import Image, Points, Surface
from napari.utils.events import EmitterGroup, Event
from napari.layers.utils.layer_utils import features_to_pandas_dataframe
from vispy.geometry import create_sphere

from .._backend import ThreeDeeModel
from ..mouse_callbacks import add_point_on_plane
from ..utils.napari_utils import add_mouse_callback_safe, \
    remove_mouse_callback_safe
from .io import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY


class SphereAnnotatorMode(Enum):
    ADD = auto()
    EDIT = auto()


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
    ANNOTATION_TYPE: str = "sphere"

    # column names in points layer features
    SPHERE_ID_FEATURES_KEY: str = "sphere_id"
    SPHERE_RADIUS_FEATURES_KEY: str = "radius"

    # metadata
    SPHERE_MESH_METADATA_KEY: str = "mesh_data"

    # parameter defaults
    DEFAULT_SPHERE_RADIUS = 5

    def __init__(
        self,
        viewer: napari.Viewer,
        image_layer: Optional[Image] = None,
        enabled: bool = False
    ):
        self.events = EmitterGroup(
            source=self,
            current_sphere_id=Event
        )

        self.viewer = viewer
        self.image_layer = image_layer
        self.points_layer = None
        self.surface_layer = None
        self.enabled = enabled
        self.mode = SphereAnnotatorMode.ADD

        if image_layer is not None:
            self.set_layers(self.image_layer)

    @property
    def active_sphere_id(self) -> Union[int, None]:
        if self.points_layer is None:
            return None
        elif self.points_layer.selected_data != {}:
            return int(list(self.points_layer.selected_data)[0])
        else:
            return None

    @property
    def active_sphere_center(self) -> np.ndarray:
        return self.points_layer.data[self._active_sphere_index]

    @property
    def active_sphere_radius(self) -> Union[float, None]:
        df = features_to_pandas_dataframe(self.points_layer.features)
        if len(df) == 0:
            return None
        else:
            return float(df[self.SPHERE_RADIUS_FEATURES_KEY].iloc[self._active_sphere_index])

    @property
    def _active_sphere_index(self) -> Union[int, None]:
        """index into data/features of current sphere"""
        if self.active_sphere_id is None:
            return None
        df = features_to_pandas_dataframe(self.points_layer.features)
        idx = df[self.SPHERE_ID_FEATURES_KEY] == self.active_sphere_id
        df = df.loc[idx, :]
        return df.index.values[0]

    @property
    def mode(self) -> SphereAnnotatorMode:
        return self._mode

    @mode.setter
    def mode(self, value: SphereAnnotatorMode):
        self._mode = value
        if self._mode == SphereAnnotatorMode.ADD:
            if self.points_layer is None:
                sphere_ids = []
            else:
                sphere_ids = self.points_layer.features[self.SPHERE_ID_FEATURES_KEY]
                self.points_layer.selected_data = {}
            if len(sphere_ids) == 0:
                new_sphere_id = 1
            else:
                new_sphere_id = np.max(sphere_ids) + 1
            self._update_current_properties(sphere_id=new_sphere_id)

    def _enable_add_mode(self, event=None):
        """Callback for enabling add mode."""
        self.mode = SphereAnnotatorMode.ADD

    def _mouse_callback(self, viewer, event):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        replace_selected = True if self.mode == SphereAnnotatorMode.EDIT else False
        add_point_on_plane(
            viewer=viewer,
            event=event,
            points_layer=self.points_layer,
            plane_layer=self.image_layer,
            replace_selected=replace_selected,
        )
        self.mode = SphereAnnotatorMode.EDIT

    def _set_radius_from_mouse_event(self, event: Event = None):
        # early exits
        if (self.image_layer is None) or (self.points_layer is None):
            return
        if not self.image_layer.visible or self.active_sphere_center is None:
            return
        if list(self.points_layer.selected_data) == []:
            return

        # Calculate intersection of click with plane through data in displayed data (scene) coordinates
        displayed_dims = np.asarray(self.viewer.dims.displayed)[
            list(self.viewer.dims.displayed_order)]
        cursor_position_3d = np.asarray(self.viewer.cursor.position)[displayed_dims]
        intersection_3d = self.image_layer.plane.intersect_with_line(
            line_position=cursor_position_3d,
            line_direction=self.viewer.camera.view_direction
        )
        current_position_3d = self.active_sphere_center[displayed_dims]
        radius = np.linalg.norm(current_position_3d - intersection_3d)
        self._update_active_sphere_radius(radius=radius)
        self._update_current_properties(radius=radius)
        self._update_spheres()

    def _update_active_sphere_radius(self, radius: float):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.points_layer.features[self.SPHERE_RADIUS_FEATURES_KEY].iloc[self._active_sphere_index] = radius

    def _update_current_properties(
        self,
        sphere_id: Optional[int] = None,
        radius: Optional[float] = None
    ):
        if self.points_layer is None:
            return
        if sphere_id is None:
            sphere_id = self.points_layer.current_properties[self.SPHERE_ID_FEATURES_KEY][0]
        if radius is None:
            radius = self.points_layer.current_properties[self.SPHERE_RADIUS_FEATURES_KEY][0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.points_layer.current_properties = {
                self.SPHERE_ID_FEATURES_KEY: [sphere_id],
                self.SPHERE_RADIUS_FEATURES_KEY: [radius]
            }

    def _create_points_layer(self) -> Optional[Points]:
        ndim = self.image_layer.data.ndim if self.image_layer is not None else 3
        layer = Points(
            data=[0] * ndim,
            ndim=ndim,
            name="sphere centers",
            size=7,
            features={
                self.SPHERE_ID_FEATURES_KEY: [0],
                self.SPHERE_RADIUS_FEATURES_KEY: [self.DEFAULT_SPHERE_RADIUS]},
            face_color=self.SPHERE_ID_FEATURES_KEY,
            face_color_cycle=self.COLOR_CYCLE,
            metadata={
                N3D_METADATA_KEY: {
                    ANNOTATION_TYPE_KEY: self.ANNOTATION_TYPE,
                    self.SPHERE_MESH_METADATA_KEY: None}
            }
        )
        layer.selected_data = {0}
        layer.remove_selected()
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
            self.viewer.bind_key(
                'r', self._set_radius_from_mouse_event, overwrite=True
            )
            self.viewer.bind_key(
                'n', self._enable_add_mode, overwrite=True
            )
            self.viewer.layers.selection.active = self.image_layer

    def _on_disable(self):
        remove_mouse_callback_safe(
            callback_list=self.viewer.mouse_drag_callbacks,
            callback=self._mouse_callback
        )
        if self.points_layer is not None:
            self.points_layer.events.data.disconnect(self._on_point_data_changed)
        self.viewer.bind_key('n', None, overwrite=True)

    def _on_point_data_changed(self, event=None):
        self._update_spheres()

    def _update_spheres(self):
        n3d_metadata = self.points_layer.metadata[N3D_METADATA_KEY]
        n3d_metadata[self.SPHERE_MESH_METADATA_KEY] = None
        grouped_points_features = self.points_layer.features.groupby(self.SPHERE_ID_FEATURES_KEY)
        sphere_vertices = []
        sphere_faces = []
        face_index_offset = 0
        for sphere_id, sphere_df in grouped_points_features:
            point_index = int(sphere_df.index.values[0])
            position = self.points_layer.data[point_index]
            radius = float(sphere_df[self.SPHERE_RADIUS_FEATURES_KEY].iloc[0])
            mesh_data = create_sphere(radius=radius, rows=20, cols=20)
            vertex_data = mesh_data.get_vertices() + position
            sphere_vertices.append(vertex_data)
            sphere_faces.append(mesh_data.get_faces() + face_index_offset)
            face_index_offset += len(vertex_data)
        if len(sphere_vertices) > 0:
            sphere_vertices = np.concatenate(sphere_vertices, axis=0)
            sphere_faces = np.concatenate(sphere_faces, axis=0)
            n3d_metadata[self.SPHERE_MESH_METADATA_KEY] = (
                sphere_vertices, sphere_faces
            )
            self._draw_spheres()
        else:
            n3d_metadata[self.SPHERE_MESH_METADATA_KEY] = None

    def _draw_spheres(self):
        n3d_metadata = self.points_layer.metadata[N3D_METADATA_KEY]
        if self.surface_layer is None:
            self.surface_layer = self._create_surface_layer()
            self.viewer.layers.append(self.surface_layer)
        else:
            self.surface_layer.data = n3d_metadata[self.SPHERE_MESH_METADATA_KEY]
