import os

import napari
import numpy as np
import zarr
from pydantic import validator

from napari_threedee.annotators.base import N3dDataModel
from napari_threedee.annotators.constants import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY
from napari_threedee.annotators.spheres.constants import SPHERE_RADIUS_FEATURES_KEY, \
    SPHERE_ANNOTATION_TYPE_KEY, SPHERE_ID_FEATURES_KEY, COLOR_CYCLE
from napari_threedee.annotators.spheres.validation import validate_layer, \
    validate_n3d_zarr


class N3dSpheres(N3dDataModel):
    centers: np.ndarray
    radii: np.ndarray

    @classmethod
    def from_layer(cls, layer: napari.layers.Points):
        centers = np.array(layer.data, dtype=np.float32)
        radii = np.array(layer.features[SPHERE_RADIUS_FEATURES_KEY], dtype=np.float32)
        return cls(centers=centers, radii=radii)

    def as_layer(self) -> napari.layers.Points:
        metadata = {N3D_METADATA_KEY: {ANNOTATION_TYPE_KEY: SPHERE_ANNOTATION_TYPE_KEY}}
        features = {
            SPHERE_RADIUS_FEATURES_KEY: self.radii,
            SPHERE_ID_FEATURES_KEY: np.arange(len(self.centers))
        }
        layer = napari.layers.Points(
            data=self.centers,
            name="n3d spheres",
            metadata=metadata,
            features=features,
            face_color=SPHERE_ID_FEATURES_KEY,
            face_color_cycle=COLOR_CYCLE,
        )
        validate_layer(layer)
        return layer

    @classmethod
    def from_n3d_zarr(cls, path: os.PathLike):
        n3d_zarr = zarr.open(path)
        validate_n3d_zarr(n3d_zarr)
        centers = np.array(n3d_zarr, dtype=np.float32)
        radii = np.array(n3d_zarr.attrs[SPHERE_RADIUS_FEATURES_KEY], dtype=np.float32)
        return cls(centers=centers, radii=radii)

    def to_n3d_zarr(self, path: os.PathLike) -> None:
        n3d_zarr = zarr.open_array(
            store=path,
            shape=self.centers.shape,
            dtype=self.centers.dtype,
            mode="w"
        )
        n3d_zarr[...] = self.centers
        n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = SPHERE_ANNOTATION_TYPE_KEY
        n3d_zarr.attrs[SPHERE_RADIUS_FEATURES_KEY] = list(self.radii)

    @validator('centers', 'radii', pre=True)
    def ensure_float32_ndarray(cls, value):
        return np.asarray(value, dtype=np.float32)

    @validator('centers')
    def ensure_at_least_2d(cls, value):
        return np.atleast_2d(value)
