import warnings
from enum import Enum, auto
from typing import Optional, Union

import napari
import numpy as np
from napari.layers import Image, Points, Surface
from napari.utils.events import EmitterGroup, Event
from napari.layers.utils.layer_utils import features_to_pandas_dataframe

from napari_threedee._backend import N3dComponent
from napari_threedee.annotators.spheres.constants import SPHERE_ID_FEATURES_KEY, \
    SPHERE_RADIUS_FEATURES_KEY, SPHERE_MESH_METADATA_KEY
from napari_threedee.utils.mouse_callbacks import on_mouse_alt_click_add_point_on_plane
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, \
    remove_mouse_callback_safe, add_point_on_plane
from napari_threedee.annotators.constants import N3D_METADATA_KEY


class SphereAnnotatorMode(Enum):
    ADD = auto()
    EDIT = auto()


class SphereAnnotator(N3dComponent):
    def __init__(
        self,
        viewer: napari.Viewer,
        image_layer: Optional[Image] = None,
        points_layer: Optional[Points] = None,
        enabled: bool = False
    ):
        self.events = EmitterGroup(
            source=self,
            current_sphere_id=Event
        )

        self.viewer = viewer
        self.image_layer = image_layer
        self.points_layer = points_layer
        self.surface_layer = None
        if self.points_layer is not None:
            self._update_spheres()
        self.enabled = enabled
        self.mode = SphereAnnotatorMode.ADD

        if image_layer is not None:
            self.set_layers(self.image_layer)

    @property
    def active_sphere_id(self) -> Union[int, None]:
        if self.points_layer is None:
            return None
        elif list(self.points_layer.selected_data) != []:
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
            radius = df[SPHERE_RADIUS_FEATURES_KEY].iloc[self._active_sphere_index]
            return float(radius)

    @property
    def _active_sphere_index(self) -> Union[int, None]:
        """index into data/features of current sphere"""
        if self.points_layer is None:
            return None
        elif list(self.points_layer.selected_data) != []:
            return int(list(self.points_layer.selected_data)[0])
        else:
            return None

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
                sphere_ids = self.points_layer.features[SPHERE_ID_FEATURES_KEY]
                self.points_layer.selected_data = {}
            if len(sphere_ids) == 0:
                new_sphere_id = 1
            else:
                new_sphere_id = np.max(sphere_ids) + 1
            self._update_current_properties(sphere_id=new_sphere_id)

    def _add_point_on_mouse_alt_click(self, viewer, event):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        if ('Alt' not in event.modifiers):
            return
        replace_selected = True if self.mode == SphereAnnotatorMode.EDIT else False
        with self.points_layer.events.highlight.blocker():
            on_mouse_alt_click_add_point_on_plane(
                viewer=viewer,
                event=event,
                points_layer=self.points_layer,
                image_layer=self.image_layer,
                replace_selected=replace_selected,
            )
        self.mode = SphereAnnotatorMode.EDIT

    def _add_point_on_key_press(self, *args):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        add_point_on_plane(
            viewer=self.viewer,
            image_layer=self.image_layer,
            points_layer=self.points_layer,
        )

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
            self.points_layer.features[SPHERE_RADIUS_FEATURES_KEY].iloc[
                self._active_sphere_index
            ] = radius

    def _update_current_properties(
        self,
        sphere_id: Optional[int] = None,
        radius: Optional[float] = None
    ):
        if self.points_layer is None:
            return
        if sphere_id is None:
            sphere_id = self.points_layer.current_properties[SPHERE_ID_FEATURES_KEY][0]
        if radius is None:
            radius = self.points_layer.current_properties[SPHERE_RADIUS_FEATURES_KEY][0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.points_layer.current_properties = {
                SPHERE_ID_FEATURES_KEY: [sphere_id],
                SPHERE_RADIUS_FEATURES_KEY: [radius]
            }

    def _create_points_layer(self) -> Optional[Points]:
        from napari_threedee.data_models.spheres import N3dSpheres
        ndim = self.image_layer.data.ndim if self.image_layer is not None else 3
        layer = N3dSpheres(centers=[0] * ndim, radii=[0]).as_layer()
        layer.selected_data = {0}
        layer.remove_selected()
        return layer

    def _create_surface_layer(self) -> Surface:
        return Surface(
            data=(np.array([[0, 0, 0]]), np.array([[0, 0, 0]])),
            name="sphere meshes",
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
                callback=self._add_point_on_mouse_alt_click
            )
            self.image_layer.bind_key('a', self._add_point_on_key_press)
            self.points_layer.events.data.connect(self._on_point_data_changed)
            self.points_layer.events.highlight.connect(self._on_highlight_change)
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
            callback=self._add_point_on_mouse_alt_click
        )
        if self.points_layer is not None:
            self.points_layer.events.data.disconnect(self._on_point_data_changed)
        self.viewer.bind_key('n', None, overwrite=True)
        self.viewer.bind_key('r', None, overwrite=True)

    def _on_point_data_changed(self, event=None):
        self._update_spheres()

    def _update_spheres(self):
        from napari_threedee.data_models import N3dSpheres
        vertices, faces = N3dSpheres.from_layer(self.points_layer).to_mesh()
        n3d_metadata = self.points_layer.metadata[N3D_METADATA_KEY]
        if len(vertices) > 0:
            n3d_metadata[SPHERE_MESH_METADATA_KEY] = (vertices, faces)
            self._draw_spheres()
        else:
            n3d_metadata[SPHERE_MESH_METADATA_KEY] = None

    def _draw_spheres(self):
        n3d_metadata = self.points_layer.metadata[N3D_METADATA_KEY]
        if self.surface_layer is None:
            self.surface_layer = self._create_surface_layer()
        if self.surface_layer not in self.viewer.layers:
            self.viewer.layers.append(self.surface_layer)
        self.surface_layer.data = n3d_metadata[SPHERE_MESH_METADATA_KEY]

    def _enable_add_mode(self, event=None):
        """Callback for enabling add mode."""
        self.mode = SphereAnnotatorMode.ADD

    def _on_highlight_change(self, event=None):
        """Callback for enabling edit mode."""
        self.mode = SphereAnnotatorMode.EDIT
