import os

import napari.layers
import numpy as np
import zarr
from pydantic import validator

from napari_threedee.annotators.base import N3dDataModel
from napari_threedee.annotators.constants import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY
from napari_threedee.annotators.points.constants import POINT_ANNOTATION_TYPE_KEY
from napari_threedee.annotators.points.validation import (
    validate_layer,
    validate_n3d_zarr,
)


class N3dPoints(N3dDataModel):
    data: np.ndarray

    @classmethod
    def from_layer(cls, layer: napari.layers.Points):
        return cls(data=layer.data)

    def as_layer(self) -> napari.layers.Points:
        if len(self.data) == 0:  # workaround for napari/napari#4213
            cls = type(self)
            n3d_points = cls(data=[0, 0, 0])
            layer = n3d_points.as_layer()
            layer.selected_data = {0}
            layer.remove_selected()
            return layer
        metadata = {N3D_METADATA_KEY: {ANNOTATION_TYPE_KEY: POINT_ANNOTATION_TYPE_KEY}}
        layer = napari.layers.Points(
            data=self.data,
            metadata=metadata,
            name='n3d points',
            ndim=self.data.shape[-1],
        )
        validate_layer(layer)
        return layer

    @classmethod
    def from_n3d_zarr(cls, path: os.PathLike):
        n3d_zarr = zarr.open(path)
        validate_n3d_zarr(n3d_zarr)
        return cls(data=np.array(n3d_zarr))

    def to_n3d_zarr(self, path: os.PathLike) -> None:
        n3d_zarr = zarr.open_array(
            store=path,
            shape=self.data.shape,
            dtype=self.data.dtype,
            mode="w"
        )
        n3d_zarr[...] = self.data
        n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = POINT_ANNOTATION_TYPE_KEY

    @validator('data', pre=True)
    def ensure_2d_float32_array(cls, value):
        data = np.atleast_2d(np.asarray(value, dtype=np.float32))
        if data.shape[-1] == 0:
            data = np.zeros(shape=(0, 3))
        return data
