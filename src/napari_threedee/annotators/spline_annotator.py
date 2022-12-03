
import einops

import napari
from napari.layers import Image, Points, Shapes
from napari.utils.events.event import EmitterGroup, Event
import numpy as np
from psygnal import EventedModel
from pydantic import validator, PrivateAttr
from scipy.interpolate import splprep, splev

from napari_threedee._backend.threedee_model import ThreeDeeModel
from ..mouse_callbacks import add_point_on_plane
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, remove_mouse_callback_safe

from typing import Tuple, Union, Optional


class _NDimensionalFilament(EventedModel):
    points: np.ndarray
    _n_spline_samples = 10000
    _raw_spline_tck = PrivateAttr(Tuple)
    _equidistant_spline_tck = PrivateAttr(Tuple)
    _length = PrivateAttr(float)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prepare_splines()

    @property
    def _ndim(self) -> int:
        return self.points.shape[-1]

    @validator('points')
    def is_coordinate_array(cls, v):
        points = np.atleast_2d(np.array(v))
        if points.ndim != 2:
            raise ValueError('must be an (n, d) array')
        return points

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'points':  # ensure splines stay in sync
            self._prepare_splines()

    def _prepare_splines(self):
        self._calculate_filament_spline_parameters()
        self._calculate_equidistant_spline_parameters()

    def _calculate_filament_spline_parameters(self):
        """Spline parametrisation mapping [0, 1] to a smooth curve through filament points.
        Note: equidistant sampling of this spline parametrisation will not yield equidistant
        samples in Euclidean space.
        """
        self._raw_spline_tck, _ = splprep(self.points.T, s=0, k=3)

    def _calculate_equidistant_spline_parameters(self):
        """Calculate a mapping of normalised cumulative distance to linear samples range [0, 1].
        * Normalised cumulative distance is the cumulative euclidean distance along the filament
          rescaled to a range of [0, 1].
        * The spline parametrisation calculated here can be used to map linearly spaced values
        which when used in the filament spline parametrisation, yield equidistant points in
        Euclidean space.
        """
        # sample the current filament spline parametrisation, yielding non-equidistant samples
        u = np.linspace(0, 1, self._n_spline_samples)
        filament_samples = splev(u, self._raw_spline_tck)
        filament_samples = np.stack(filament_samples, axis=1)

        # calculate the cumulative length of line segments as we move along the filament.
        inter_point_differences = np.diff(filament_samples, axis=0)
        # inter_point_differences = np.r_[np.zeros((1, 3)), inter_point_differences]  # prepend a row of zeros
        inter_point_distances = np.linalg.norm(inter_point_differences, axis=1)
        cumulative_distance = np.cumsum(inter_point_distances)

        # calculate spline mapping normalised cumulative distance to linear samples in [0, 1]
        self._length = cumulative_distance[-1]
        cumulative_distance /= self._length
        # prepend a zero, no distance has been covered at start of spline parametrisation
        cumulative_distance = np.r_[[0], cumulative_distance]
        self._equidistant_spline_tck, _ = splprep([u], u=cumulative_distance, s=0, k=3)

    def _sample_backbone(self, u: Union[float, np.ndarray], derivative: int = 0):
        """Sample points or derivatives along the backbone of the filament.
        This function
        * maps values in the range [0, 1] to points on the smooth filament backbone.
        * yields equidistant samples along the filament for linearly spaced values of u.
        If calculate_derivate=True then the derivative will be evaluated and returned instead of
        backbone points.
        """
        u = splev([np.asarray(u)], self._equidistant_spline_tck)  # [
        samples = splev(u, self._raw_spline_tck, der=derivative)
        return einops.rearrange(samples, 'c 1 1 b -> b c')

    def _get_equidistant_u(self, separation: float) -> np.ndarray:
        """Get values for u which yield backbone samples with a defined Euclidean separation."""
        n_points = int(self._length // separation)
        remainder = (self._length % separation) / self._length
        return np.linspace(0, 1 - remainder, n_points)

    def _get_equidistant_backbone_samples(
            self, separation: float, calculate_derivative: bool = False
    ) -> np.ndarray:
        """Calculate equidistant backbone samples with a defined separation in Euclidean space."""
        u = self._get_equidistant_u(separation)
        return self._sample_backbone(u, derivative=calculate_derivative)


class SplineAnnotator(ThreeDeeModel):
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
    # column name in the control points layer features to store the spline ID
    SPLINE_ID_COLUMN = "spline_id"
    SPLINE_ORDER = 3

    def __init__(
            self,
            viewer: napari.Viewer,
            image_layer: Optional[Image] = None,
            enabled: bool = False
    ):
        self.events = EmitterGroup(
            source=self,
            current_spline_id=Event,
            splines_updated=Event,
        )

        self.viewer = viewer
        self.image_layer = image_layer
        self.points_layer = None
        self.shapes_layer = None
        self.auto_fit_spline = True
        self.enabled = enabled

        self.current_spline_id: int = 0

        # storage for the spline objects
        # each spline is in its own object
        self._splines = dict()

        if image_layer is not None:
            self.set_layers(self.image_layer)

    @property
    def current_spline_id(self):
        return self._current_spline_id

    @current_spline_id.setter
    def current_spline_id(self, id: int):
        self._current_spline_id = np.clip(id, 0, None)
        if self.points_layer is not None:
            self.points_layer.selected_data = {}
            self.points_layer.current_properties = {
                self.SPLINE_ID_COLUMN: self.current_spline_id
            }
        self.events.current_spline_id()

    def next_spline(self, event=None):
        self.current_spline_id += 1

    def previous_spline(self, event=None):
        self.current_spline_id -= 1

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
            name="spline control points",
            size=3,
            features={self.SPLINE_ID_COLUMN: [0]},
            face_color=self.SPLINE_ID_COLUMN,
            face_color_cycle=self.COLOR_CYCLE,
            metadata={"splines": dict()}
        )
        layer.selected_data = {0}
        layer.remove_selected()
        self.current_spline_id = self.current_spline_id
        return layer

    def _create_shapes_layer(self) -> Shapes:
        return Shapes(ndim=self.image_layer.data.ndim, name="splines", edge_color="green")

    def set_layers(self, image_layer: napari.layers.Image):
        self.image_layer = image_layer
        if self.points_layer is None and self.image_layer is not None:
            self.points_layer = self._create_points_layer()
            self.viewer.add_layer(self.points_layer)
            self.shapes_layer = self._create_shapes_layer()
            self.viewer.add_layer(self.shapes_layer)

    def _on_enable(self):
        if self.points_layer is not None:
            add_mouse_callback_safe(
                self.viewer.mouse_drag_callbacks, self._mouse_callback
            )
            self.points_layer.events.data.connect(self._on_point_data_changed)
            self.viewer.bind_key('n', self.next_spline)
            self.viewer.layers.selection.active = self.image_layer

    def _on_disable(self):
        remove_mouse_callback_safe(
            self.viewer.mouse_drag_callbacks, self._mouse_callback
        )
        if self.points_layer is not None:
            self.points_layer.events.data.disconnect(self._on_point_data_changed)
        self.viewer.bind_key('n', None)

    def _on_point_data_changed(self, event=None):
        if self.auto_fit_spline is True:
            self._update_splines()
            self._draw_splines()

    def _update_splines(self):
        grouped_points_features = self.points_layer.features.groupby(self.SPLINE_ID_COLUMN)
        splines = dict()
        for spline_name, spline_df in grouped_points_features:
            point_indices = spline_df.index.tolist()
            if len(point_indices) > self.SPLINE_ORDER:
                # the number of points must be greater than the spline order to properly fit
                spline_coordinates = self.points_layer.data[point_indices]
                splines[spline_name] = _NDimensionalFilament(points=spline_coordinates, k=self.SPLINE_ORDER)
        self.points_layer.metadata["splines"] = splines
        self.events.splines_updated()

    def _clear_shapes_layer(self):
        """Delete all shapes in the shapes layer."""
        if self.shapes_layer is None:
            return
        n_shapes = len(self.shapes_layer.data)
        self.shapes_layer.selected_data = set(np.arange(n_shapes))
        self.shapes_layer.remove_selected()

    def _draw_splines(self):
        self._clear_shapes_layer()
        for spline_name, spline_object in self.points_layer.metadata["splines"].items():
            spline_points = spline_object._sample_backbone(u=np.linspace(0, 1, 50))
            self.shapes_layer.add_paths(spline_points)
