import napari
from napari.layers import Image, Points, Shapes
from napari.utils.events.event import EmitterGroup, Event
import numpy as np
from typing import Optional, Dict

from napari_threedee._backend.threedee_model import N3dComponent
from napari_threedee.annotators.paths.constants import PATH_ANNOTATION_TYPE_KEY, \
    COLOR_CYCLE, PATH_ID_FEATURES_KEY, PATH_COLOR_FEATURES_KEY, \
    SPLINES_METADATA_KEY, SPLINE_ORDER
from napari_threedee.annotators.paths.sampler import SplineSampler
from napari_threedee.mouse_callbacks import add_point_on_plane
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, \
    remove_mouse_callback_safe
from napari_threedee.annotators.constants import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY


class PathAnnotator(N3dComponent):
    def __init__(
        self,
        viewer: napari.Viewer,
        image_layer: Optional[Image] = None,
        points_layer: Optional[Points] = None,
        enabled: bool = False
    ):
        self.events = EmitterGroup(
            source=self,
            active_spline_id=Event,
            splines_updated=Event,
        )

        self.viewer = viewer
        self._active_spline_id: int = 0

        self.image_layer = image_layer
        self.points_layer = points_layer
        self.shapes_layer = None

        self.auto_fit_spline = True

        # storage for the spline objects
        # each spline is in its own object
        self._splines = dict()

        if image_layer is not None:
            self.set_layers(self.image_layer)

        self.enabled = enabled

    @property
    def active_spline_id(self):
        return self._active_spline_id

    @active_spline_id.setter
    def active_spline_id(self, id: int):
        self._active_spline_id = np.clip(id, 0, None)
        if self.points_layer is not None:
            self.points_layer.selected_data = {}
            self.points_layer.current_properties = {
                PATH_ID_FEATURES_KEY: self.active_spline_id
            }
        self.events.active_spline_id()

    def next_spline(self, event=None):
        self.active_spline_id += 1

    def previous_spline(self, event=None):
        self.active_spline_id -= 1

    def _mouse_callback(self, viewer, event):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        add_point_on_plane(
            viewer=viewer,
            event=event,
            points_layer=self.points_layer,
            plane_layer=self.image_layer
        )

    def _create_points_layer(self) -> Optional[Points]:
        layer = Points(
            data=[0] * self.image_layer.data.ndim,
            ndim=self.image_layer.data.ndim,
            name="n3d paths (control points)",
            size=3,
            features={PATH_ID_FEATURES_KEY: [0]},
            face_color=PATH_ID_FEATURES_KEY,
            face_color_cycle=COLOR_CYCLE,
            metadata={
                N3D_METADATA_KEY: {
                    ANNOTATION_TYPE_KEY: PATH_ANNOTATION_TYPE_KEY,
                    SPLINES_METADATA_KEY: dict,
                }
            }
        )
        layer.selected_data = {0}
        layer.remove_selected()
        self.active_spline_id = self.active_spline_id
        return layer

    def _create_shapes_layer(self) -> Shapes:
        return Shapes(
            ndim=self.image_layer.data.ndim,
            name="n3d paths (smooth fit)",
            edge_color="green"
        )

    def set_layers(self, image_layer: napari.layers.Image):
        self.image_layer = image_layer
        if self.image_layer is not None:
            if self.points_layer is None:
                self.points_layer = self._create_points_layer()
            if self.points_layer not in self.viewer.layers:
                self.viewer.add_layer(self.points_layer)
            if self.shapes_layer is None:
                self.shapes_layer = self._create_shapes_layer()
            if self.shapes_layer not in self.viewer.layers:
                self.viewer.add_layer(self.shapes_layer)
            self._update_splines()
            self._draw_splines()

    def _on_enable(self):
        if self.points_layer is not None:
            add_mouse_callback_safe(
                self.viewer.mouse_drag_callbacks, self._mouse_callback
            )
            self.points_layer.events.data.connect(self._on_point_data_changed)
            self.viewer.bind_key('n', self.next_spline, overwrite=True)
            self.viewer.layers.selection.active = self.image_layer

    def _on_disable(self):
        remove_mouse_callback_safe(
            self.viewer.mouse_drag_callbacks, self._mouse_callback
        )
        if self.points_layer is not None:
            self.points_layer.events.data.disconnect(
                self._on_point_data_changed
            )
        self.viewer.bind_key('n', None, overwrite=True)

    def _on_point_data_changed(self, event=None):
        if self.auto_fit_spline is True:
            self._update_splines()
            self._draw_splines()

    def _update_splines(self):
        grouped_points_features = self.points_layer.features.groupby(
            PATH_ID_FEATURES_KEY
        )
        splines = dict()
        for spline_name, spline_df in grouped_points_features:
            point_indices = spline_df.index.tolist()
            if len(point_indices) > SPLINE_ORDER:
                # the number of points must be greater than the spline order to properly fit
                spline_coordinates = self.points_layer.data[point_indices]
                splines[spline_name] = SplineSampler(
                    points=spline_coordinates, k=SPLINE_ORDER
                )
        metadata = {
            SPLINES_METADATA_KEY: splines
        }
        self.points_layer.metadata[N3D_METADATA_KEY].update(metadata)
        self.events.splines_updated()

    def _get_spline_colors(self) -> Dict[int, np.ndarray]:
        self.points_layer.features[PATH_COLOR_FEATURES_KEY] = \
            list(self.points_layer.face_color)
        grouped_points_features = self.points_layer.features.groupby(
            PATH_ID_FEATURES_KEY
        )
        spline_colors = dict()
        for spline_id, spline_df in grouped_points_features:
            spline_colors[spline_id] = spline_df[PATH_COLOR_FEATURES_KEY].iloc[0]
        return spline_colors

    def _clear_shapes_layer(self):
        """Delete all shapes in the shapes layer."""
        if self.shapes_layer is None:
            return
        n_shapes = len(self.shapes_layer.data)
        self.shapes_layer.selected_data = set(np.arange(n_shapes))
        self.shapes_layer.remove_selected()

    def _draw_splines(self):
        self._clear_shapes_layer()
        splines = self.points_layer.metadata[N3D_METADATA_KEY][SPLINES_METADATA_KEY]
        spline_colors = self._get_spline_colors()
        for spline_id, spline_object in splines.items():
            spline_points = spline_object._sample_backbone(
                u=np.linspace(0, 1, 1000)
            )
            spline_color = spline_colors[spline_id]
            self.shapes_layer.add_paths(spline_points, edge_color=spline_color)