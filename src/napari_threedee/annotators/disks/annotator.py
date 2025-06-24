import warnings
from enum import Enum, auto
from typing import Optional, Union

import napari
import numpy as np
from napari.layers import Image, Points, Surface
from napari.utils.events import EmitterGroup, Event
from napari.layers.utils.layer_utils import features_to_pandas_dataframe

from napari_threedee._backend import N3dComponent
from napari_threedee.annotators.disks.constants import (
    N3D_METADATA_KEY,
    ANNOTATION_TYPE_KEY,
    DISK_ID_FEATURES_KEY,
    DISK_ANNOTATION_TYPE,
    DISK_RADIUS_FEATURES_KEY
)
from napari_threedee.manipulators.constants import ADD_POINT_KEY
from napari_threedee.utils.mouse_callbacks import on_mouse_alt_click_add_point_on_plane
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, \
    remove_mouse_callback_safe, add_point_on_plane
from napari_threedee.annotators.constants import N3D_METADATA_KEY


class DiskAnnotatorMode(Enum):
    ADD = auto()
    EDIT = auto()


class DiskAnnotator(N3dComponent):
    def __init__(
        self,
        viewer: napari.Viewer,
        image_layer: Optional[Image] = None,
        points_layer: Optional[Points] = None,
        enabled: bool = False
    ):
        self.events = EmitterGroup(
            source=self,
            current_disk_id=Event
        )

        self.viewer = viewer
        self.image_layer = image_layer
        self.points_layer = points_layer
        self.vectors_layer = None
        if self.points_layer is not None:
            self._update_disks()
        self.enabled = enabled
        self.mode = DiskAnnotatorMode.ADD

        if image_layer is not None:
            self.set_layers(self.image_layer)

    @property
    def active_disk_id(self) -> Union[int, None]:
        if self.points_layer is None:
            return None
        elif list(self.points_layer.selected_data) != []:
            return int(list(self.points_layer.selected_data)[0])
        else:
            return None

    @property
    def active_disk_position(self) -> np.ndarray:
        return self.points_layer.data[self._active_disk_index]

    @property
    def active_disk_radius(self) -> Union[float, None]:
        df = features_to_pandas_dataframe(self.points_layer.features)
        if len(df) == 0:
            return None
        radius = df[DISK_RADIUS_FEATURES_KEY].iloc[self._active_disk_index]
        return float(radius)

    @property
    def _active_disk_index(self) -> Union[int, None]:
        """index into data/features of current dipole"""
        if self.points_layer is None:
            return None
        elif list(self.points_layer.selected_data) != []:
            return int(list(self.points_layer.selected_data)[0])
        else:
            return None

    @property
    def mode(self) -> DiskAnnotatorMode:
        return self._mode

    @mode.setter
    def mode(self, value: DiskAnnotatorMode):
        self._mode = value
        if self._mode == DiskAnnotatorMode.ADD:
            if self.points_layer is None:
                dipole_ids = []
            else:
                dipole_ids = self.points_layer.features[DISK_ID_FEATURES_KEY]
                self.points_layer.selected_data = {}
            if len(dipole_ids) == 0:
                new_dipole_id = 1
            else:
                new_dipole_id = np.max(dipole_ids) + 1
            self._update_current_properties(disk_id=new_dipole_id)

    def _add_point_on_mouse_alt_click(self, viewer, event):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        if ('Alt' not in event.modifiers):
            return
        replace_selected = True if self.mode == DiskAnnotatorMode.EDIT else False
        with self.points_layer.events.highlight.blocker():
            on_mouse_alt_click_add_point_on_plane(
                viewer=viewer,
                event=event,
                points_layer=self.points_layer,
                image_layer=self.image_layer,
                replace_selected=replace_selected,
            )
        self.mode = DiskAnnotatorMode.EDIT

    def _add_point_on_key_press(self, *args):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        replace_selected = True if self.mode == DiskAnnotatorMode.EDIT else False
        add_point_on_plane(
            viewer=self.viewer,
            image_layer=self.image_layer,
            points_layer=self.points_layer,
            replace_selected=replace_selected,
        )

    def _set_direction_from_mouse_event(self, event: Event = None):
        # early exits
        if (self.image_layer is None) or (self.points_layer is None):
            return
        if not self.image_layer.visible or self.active_disk_position is None:
            return
        if list(self.points_layer.selected_data) == []:
            return

        # Calculate intersection of click with plane through data in displayed data (scene) coordinates
        displayed_dims = np.asarray(self.viewer.dims.displayed)[list(self.viewer.dims.displayed_order)]
        cursor_position_3d = np.asarray(self.viewer.cursor.position)[displayed_dims]
        disk_edge_position_3d = self.image_layer.plane.intersect_with_line(
            line_position=cursor_position_3d,
            line_direction=self.viewer.camera.view_direction
        )
        disk_center_position_3d = self.active_disk_position[displayed_dims]
        vector = disk_edge_position_3d - disk_center_position_3d
        radius = np.linalg.norm(vector)
        if np.abs(radius) < 1e-10:
            return
        self._update_active_disk_radius(radius=radius)
        self._update_current_properties(direction=vector)
        self._update_disks()

    def _update_active_disk_radius(self, radius: float):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.points_layer.features[DISK_RADIUS_FEATURES_KEY].iloc[self._active_disk_index] = radius

    def _update_current_properties(
        self,
        disk_id: Optional[int] = None,
        direction: Optional[float] = None
    ):
        if self.points_layer is None:
            return
        if disk_id is None:
            disk_id = self.points_layer.current_properties[DISK_ID_FEATURES_KEY][0]
        if direction is None:
            radius = self.points_layer.current_properties[DISK_RADIUS_FEATURES_KEY][0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.points_layer.current_properties = {
                DISK_ID_FEATURES_KEY: [disk_id],
                DISK_RADIUS_FEATURES_KEY: [radius],
            }

    def _create_points_layer(self) -> Optional[Points]:
        from napari_threedee.data_models.dipoles import N3dDipoles
        ndim = self.image_layer.data.ndim if self.image_layer is not None else 3
        layer = N3dDipoles.from_centers_and_directions(centers=np.zeros((ndim, 3)),
                                                       directions=np.zeros((ndim, 3))).as_layer()
        layer.selected_data = {0}
        layer.remove_selected()
        return layer

    def _create_surface_layer(self) -> Surface:
        return Vectors(
            data=np.zeros(shape=(0, 2, 3)),
            name="dipoles",
            edge_width=5,
            length=20,
            opacity=0.7,
        )

    def set_layers(self, image_layer: napari.layers.Image):
        self.image_layer = image_layer
        if self.points_layer is None and self.image_layer is not None:
            self.points_layer = self._create_points_layer()
            self.viewer.add_layer(self.points_layer)
            self.vectors_layer = self._create_surface_layer()
            self.viewer.add_layer(self.vectors_layer)
            self.viewer.layers.selection.active = self.points_layer

    def _on_enable(self):
        if self.points_layer is not None:
            add_mouse_callback_safe(
                callback_list=self.viewer.mouse_drag_callbacks,
                callback=self._add_point_on_mouse_alt_click
            )
            self.image_layer.bind_key(ADD_POINT_KEY, self._add_point_on_key_press, overwrite=True)
            self.points_layer.events.data.connect(self._on_point_data_changed)
            self.points_layer.events.highlight.connect(self._on_highlight_change)
            self.viewer.bind_key(
                'v', self._set_direction_from_mouse_event, overwrite=True
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
        if self.image_layer is not None:
            self.image_layer.bind_key(ADD_POINT_KEY, None, overwrite=True)
        self.viewer.bind_key('n', None, overwrite=True)
        self.viewer.bind_key('v', None, overwrite=True)

    def _on_point_data_changed(self, event=None):
        self._update_disks()

    def _update_disks(self):
        from napari_threedee.data_models import N3dDipoles
        vectors = N3dDipoles.from_layer(self.points_layer).as_napari_vectors()
        n3d_metadata = self.points_layer.metadata[N3D_METADATA_KEY]
        if len(vectors) > 0:
            self._draw_dipoles(vectors)


    def _draw_dipoles(self, vectors: np.ndarray):
        if self.vectors_layer is None:
            self.vectors_layer = self._create_surface_layer()
        if self.vectors_layer not in self.viewer.layers:
            self.viewer.layers.append(self.vectors_layer)
        self.vectors_layer.data = vectors

    def _enable_add_mode(self, event=None):
        """Callback for enabling add mode."""
        self.mode = DiskAnnotatorMode.ADD

    def _on_highlight_change(self, event=None):
        """Callback for enabling edit mode."""
        self.mode = DiskAnnotatorMode.EDIT
