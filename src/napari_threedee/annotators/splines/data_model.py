import os
from typing import List

import napari
import numpy as np
import zarr
from pydantic import BaseModel, validator

from napari_threedee.annotators.base import N3dDataModel
from napari_threedee.annotators.constants import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY
from napari_threedee.annotators.splines.constants import SPLINE_ID_FEATURES_KEY, \
    SPLINE_ANNOTATION_TYPE_KEY, COLOR_CYCLE

from napari_threedee.annotators.splines.sampler import SplineSampler
from napari_threedee.annotators.splines.validation import validate_layer, \
    validate_n3d_zarr


class _N3dSpline(BaseModel):
    data: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @property
    def ndim(self) -> int:
        return self.data.shape[-1]

    def interpolate(self, n: int = 10000) -> np.ndarray:
        """Sample equidistant points between data points."""
        sampler = SplineSampler(points=self.data)
        return sampler._sample_backbone(u=np.linspace(0, 1, num=n))

    @validator('data', pre=True)
    def ensure_float32_ndarray(cls, value):
        return np.asarray(value, dtype=np.float32)


class N3dSplines(N3dDataModel):
    data: List[_N3dSpline]

    @property
    def n_points(self) -> int:
        return np.sum([len(spline.data) for spline in self.data])

    @property
    def ndim(self) -> int:
        return self.data[0].data.shape[-1]

    @property
    def spline_ids(self) -> np.ndarray:
        spline_ids = [[idx] * len(spline.data) for idx, spline in enumerate(self.data)]
        return np.concatenate(spline_ids)

    @classmethod
    def from_layer(cls, layer: napari.layers.Layer):
        grouped_points_features = layer.features.groupby(SPLINE_ID_FEATURES_KEY)
        splines = [
            _N3dSpline(data=layer.data[df.index.tolist()])
            for name, df in grouped_points_features
        ]
        return cls(data=splines)

    def as_layer(self) -> napari.layers.Points:
        data = np.concatenate([spline.data for spline in self.data])
        metadata = {
            N3D_METADATA_KEY: {
                ANNOTATION_TYPE_KEY: SPLINE_ANNOTATION_TYPE_KEY,
            }
        }
        features = {SPLINE_ID_FEATURES_KEY: self.spline_ids}
        layer = napari.layers.Points(
            data=data,
            metadata=metadata,
            features=features,
            face_color=SPLINE_ID_FEATURES_KEY,
            face_color_cycle=COLOR_CYCLE,
            name='n3d splines',
            ndim=data.shape[-1],
        )
        validate_layer(layer)
        return layer

    @classmethod
    def from_n3d_zarr(cls, path: os.PathLike):
        n3d_zarr = zarr.open(path)
        validate_n3d_zarr(n3d_zarr)
        points = np.asarray(n3d_zarr)
        spline_ids = n3d_zarr.attrs[SPLINE_ID_FEATURES_KEY]
        splines = [
            _N3dSpline(data=points[spline_ids == idx])
            for idx in np.unique(spline_ids)
        ]
        return cls(data=splines)

    def to_n3d_zarr(self, path: os.PathLike) -> None:
        n3d_zarr = zarr.open_array(
            store=path,
            shape=(self.n_points, self.ndim),
            dtype=np.float32,
            mode="w",
        )
        n3d_zarr[...] = np.concatenate([spline.data for spline in self.data])
        n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = SPLINE_ANNOTATION_TYPE_KEY
        n3d_zarr.attrs[SPLINE_ID_FEATURES_KEY] = list(self.spline_ids)
