from typing import Tuple, Union

import einops
import numpy as np
from psygnal import EventedModel
from pydantic import PrivateAttr, validator
from scipy.interpolate import splprep, splev


class SplineSampler(EventedModel):
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

    def __call__(self, u: Union[float, np.ndarray], derivative:int = 0):
        """Sample equidistant points along a spline interpolation of points.

        `u` in the range `[0, 1]` covers the path through `points[0]` to `points[-1]`.
        `derivative` is the derivative order to sample
        """
        return self._sample_backbone(u, derivative=derivative)

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
        if name == 'points':  # ensure data stay in sync
            self._prepare_splines()

    def _prepare_splines(self):
        spline_order = 3 if len(self.points) > 3 else len(self.points) - 1
        self._calculate_backbone_spline_parameters(k=spline_order)
        self._calculate_equidistant_spline_parameters(k=spline_order)

    def _calculate_backbone_spline_parameters(self, k: int):
        """Spline parametrisation mapping [0, 1] to a smooth curve through filament points.
        Note: equidistant sampling of this spline parametrisation will not yield equidistant
        samples in Euclidean space.
        """
        self._raw_spline_tck, _ = splprep(self.points.T, s=0, k=k)

    def _calculate_equidistant_spline_parameters(self, k: int):
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
        self._equidistant_spline_tck, _ = splprep(
            [u], u=cumulative_distance, s=0, k=k
        )

    def _sample_backbone(
        self, u: Union[float, np.ndarray], derivative: int = 0
    ):
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
