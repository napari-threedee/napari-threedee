import os
from typing import List

import napari
import numpy as np
import zarr
from pydantic import BaseModel, validator

from napari_threedee.annotators.base import N3dDataModel
from napari_threedee.annotators.constants import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY
from napari_threedee.annotators.disks.validation import validate_layer, validate_n3d_zarr

from napari_threedee.annotators.disks.constants import (
    DISK_ANNOTATION_TYPE_KEY,
    DISK_ID_FEATURES_KEY,
    DISK_NORMAL_X_FEATURES_KEY,
    DISK_NORMAL_Y_FEATURES_KEY,
    DISK_NORMAL_Z_FEATURES_KEY,
    DISK_RADIUS_FEATURES_KEY,
)


class N3dDisk(BaseModel):
    center: np.ndarray
    normal: np.ndarray
    radius: float

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    @validator('center', 'normal', pre=True)
    def ensure_float32_ndarray(cls, value):
        return np.asarray(value, dtype=np.float32)

    @validator('radius', pre=True)
    def ensure_positive_radius(cls, value):
        if value <= 0:
            raise ValueError("Radius must be positive")
        return float(value)


class N3dDisks(N3dDataModel):
    data: List[N3dDisk]

    @property
    def centers(self) -> np.ndarray:
        if len(self.data) == 0:
            return np.empty((0, 3))
        return np.stack([disk.center for disk in self.data], axis=0)

    @property
    def normals(self) -> np.ndarray:
        if len(self.data) == 0:
            return np.empty((0, 3))
        return np.stack([disk.normal for disk in self.data], axis=0)

    @property
    def radii(self) -> np.ndarray:
        if len(self.data) == 0:
            return np.empty((0,))
        return np.array([disk.radius for disk in self.data])

    def as_napari_vectors(self) -> np.ndarray:
        """Generate an (n, 2, 3) array containing vectors for napari.
        
        - arr[:, 0, :] contains start points (centers) for depicted vectors
        - arr[:, 1, :] contains normal directions of depicted disks
        """
        return np.stack([self.centers, self.normals], axis=-2)

    @classmethod
    def from_centers_normals_and_radii(cls, centers: np.ndarray, normals: np.ndarray, radii: np.ndarray):
        disks = [
            N3dDisk(center=center, normal=normal, radius=radius)
            for center, normal, radius
            in zip(centers, normals, radii)
        ]
        return cls(data=disks)

    @classmethod
    def from_layer(cls, layer: napari.layers.Layer):
       centers = np.asarray(layer.data)
       normals = np.stack([layer.features[DISK_NORMAL_Z_FEATURES_KEY],
                          layer.features[DISK_NORMAL_Y_FEATURES_KEY],
                          layer.features[DISK_NORMAL_X_FEATURES_KEY]], axis=-1)
       radii = np.asarray(layer.features[DISK_RADIUS_FEATURES_KEY])
       return cls.from_centers_normals_and_radii(centers=centers, normals=normals, radii=radii)

    def as_layer(self) -> napari.layers.Points:
        if len(self.data) == 0:
            return N3dDisks.create_empty_layer()

        normals = self.normals
        features = {
            DISK_ID_FEATURES_KEY: np.arange(len(self.data)),
            DISK_NORMAL_X_FEATURES_KEY: normals[:, -1],
            DISK_NORMAL_Y_FEATURES_KEY: normals[:, -2],
            DISK_NORMAL_Z_FEATURES_KEY: normals[:, -3],
            DISK_RADIUS_FEATURES_KEY: self.radii,
        }
        metadata = {N3D_METADATA_KEY: {ANNOTATION_TYPE_KEY: DISK_ANNOTATION_TYPE_KEY}}

        # Construct the points layer from the data
        layer = napari.layers.Points(
            data=self.centers,
            features=features,
            metadata=metadata,
            name='n3d disks',
            ndim=self.centers.shape[-1],
        )
        layer.selected_data = {len(self.centers) - 1}
        validate_layer(layer)
        return layer

    
    @classmethod
    def from_n3d_zarr(cls, path: os.PathLike):
        n3d_zarr = zarr.open(path)
        validate_n3d_zarr(n3d_zarr)
        disk_data = np.asarray(n3d_zarr)  # (n, 7) - center(3), normal(3), radius(1)
        centers = disk_data[:, :3]
        normals = disk_data[:, 3:6]
        radii = disk_data[:, 6]
        disks = [
            N3dDisk(center=center, normal=normal, radius=radius)
            for center, normal, radius
            in zip(centers, normals, radii)
        ]
        return cls(data=disks)

    def to_n3d_zarr(self, path: os.PathLike) -> None:
        n3d_zarr = zarr.open_array(
            store=path,
            shape=(len(self), 7), # disk data shape: (n, 7) - center(3), normal(3), radius(1)
            dtype=np.float32,
            mode="w",
        )
        # Stack centers, normals, and radii
        disk_data = np.column_stack([
            self.centers,
            self.normals,
            self.radii
        ])
        n3d_zarr[...] = disk_data
        n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = DISK_ANNOTATION_TYPE_KEY

    @classmethod
    def create_empty_layer(cls):
        """Create an empty disk annotator points layer.

        Returns
        -------
        layer : napari.layers.Points
            The napari Points layer initialized for disk annotation.
        """

        features = {
            DISK_ID_FEATURES_KEY: np.array([0]),
            DISK_NORMAL_X_FEATURES_KEY: np.array([0]),
            DISK_NORMAL_Y_FEATURES_KEY: np.array([0]),
            DISK_NORMAL_Z_FEATURES_KEY: np.array([1]),  # Default normal pointing up
            DISK_RADIUS_FEATURES_KEY: np.array([1.0]),  # Default radius of 1.0
        }
        metadata = {N3D_METADATA_KEY: {ANNOTATION_TYPE_KEY: DISK_ANNOTATION_TYPE_KEY}}

        # workaround for napari/napari#4213
        dummy_data = np.array([[0, 0, 0]])
        layer = napari.layers.Points(
            data=dummy_data,
            features=features,
            metadata=metadata,
            name='n3d disks',
            ndim=dummy_data.shape[-1],
        )
        layer.selected_data = {0}
        layer.remove_selected()
        return layer

    def __getitem__(self, idx: int) -> N3dDisk:
        return self.data[idx]

    def __iter__(self) -> N3dDisk:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)