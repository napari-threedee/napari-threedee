import os
from typing import List

import napari
import numpy as np
import zarr
from pydantic import BaseModel, validator

from napari_threedee.annotators.base import N3dDataModel
from napari_threedee.annotators.constants import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY
from napari_threedee.annotators.dipoles.validation import validate_layer, validate_n3d_zarr

from napari_threedee.annotators.dipoles.constants import (
    DIPOLE_ANNOTATION_TYPE_KEY,
    DIPOLE_ID_FEATURES_KEY,
    DIPOLE_DIRECTION_X_FEATURES_KEY,
    DIPOLE_DIRECTION_Y_FEATURES_KEY,
    DIPOLE_DIRECTION_Z_FEATURES_KEY,
)


class N3dDipole(BaseModel):
    center: np.ndarray
    direction: np.ndarray

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    @validator('center', 'direction', pre=True)
    def ensure_float32_ndarray(cls, value):
        return np.asarray(value, dtype=np.float32)


class N3dDipoles(N3dDataModel):
    data: List[N3dDipole]

    @property
    def centers(self) -> np.ndarray:
        if len(self.data) == 0:
            return np.empty((0, 3))
        return np.stack([dipole.center for dipole in self.data], axis=0)

    @property
    def directions(self) -> np.ndarray:
        if len(self.data) == 0:
            return np.empty((0, 3))
        return np.stack([dipole.direction for dipole in self.data], axis=0)

    def as_napari_vectors(self) -> np.ndarray:
        """Generate an (n, 2, 3) array containing vectors for napari.
        
        - arr[:, 0, :] contains start points for depicted vectors
        - arr[:, 1, :] contains directions of depicted vectors
        """
        return np.stack([self.centers, self.directions], axis=-2)

    @classmethod
    def from_centers_and_directions(cls, centers: np.ndarray, directions: np.ndarray):
        dipoles = [
            N3dDipole(center=center, direction=direction)
            for center, direction
            in zip(centers, directions)
        ]
        return cls(data=dipoles)

    @classmethod
    def from_layer(cls, layer: napari.layers.Layer):
       centers = np.asarray(layer.data)
       directions = np.stack([layer.features[DIPOLE_DIRECTION_Z_FEATURES_KEY],
                             layer.features[DIPOLE_DIRECTION_Y_FEATURES_KEY],
                             layer.features[DIPOLE_DIRECTION_X_FEATURES_KEY]], axis=-1)
       return cls.from_centers_and_directions(centers=centers, directions=directions)

    def as_layer(self) -> napari.layers.Points:
        if len(self.data) == 0:
            return N3dDipoles.create_empty_layer()

        directions = self.directions
        features = {
            DIPOLE_ID_FEATURES_KEY: np.arange(len(self.data)),
            DIPOLE_DIRECTION_X_FEATURES_KEY: directions[:, -1],
            DIPOLE_DIRECTION_Y_FEATURES_KEY: directions[:, -2],
            DIPOLE_DIRECTION_Z_FEATURES_KEY: directions[:, -3],
        }
        metadata = {N3D_METADATA_KEY: {ANNOTATION_TYPE_KEY: DIPOLE_ANNOTATION_TYPE_KEY}}

        # Construct the points layer from the data
        layer = napari.layers.Points(
            data=self.centers,
            features=features,
            metadata=metadata,
            name='n3d dipoles',
            ndim=self.centers.shape[-1],
        )
        layer.selected_data = {len(self.centers) - 1}
        validate_layer(layer)
        return layer

    
    @classmethod
    def from_n3d_zarr(cls, path: os.PathLike):
        n3d_zarr = zarr.open(path)
        validate_n3d_zarr(n3d_zarr)
        vectors = np.asarray(n3d_zarr)  # (n, 2, 3)
        centers = vectors[:, 0]
        directions = vectors[:, 1]
        dipoles = [
            N3dDipole(center=center, direction=direction)
            for center, direction
            in zip(centers, directions)
        ]
        return cls(data=dipoles)

    def to_n3d_zarr(self, path: os.PathLike) -> None:
        n3d_zarr = zarr.open_array(
            store=path,
            shape=(len(self), 2, 3), # vectors shape: (n, 2, 3)
            dtype=np.float32,
            mode="w",
        )
        n3d_zarr[...] = self.as_napari_vectors()
        n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = DIPOLE_ANNOTATION_TYPE_KEY

    @classmethod
    def create_empty_layer(cls):
        """Create an empty dipole annotator points layer.

        Returns
        -------
        layer : napari.layers.Points
            The napari Points layer initialized for dipole annotation.
        """

        features = {
            DIPOLE_ID_FEATURES_KEY: np.empty(0),
            DIPOLE_DIRECTION_X_FEATURES_KEY: np.empty(0),
            DIPOLE_DIRECTION_Y_FEATURES_KEY: np.empty(0),
            DIPOLE_DIRECTION_Z_FEATURES_KEY: np.empty(0),
        }
        metadata = {N3D_METADATA_KEY: {ANNOTATION_TYPE_KEY: DIPOLE_ANNOTATION_TYPE_KEY}}

        # workaround for napari/napari#4213
        dummy_data = np.zeros((0, 3))
        layer = napari.layers.Points(
            data=dummy_data,
            features=features,
            metadata=metadata,
            name='n3d dipoles',
            ndim=dummy_data.shape[-1],
        )
        return layer

    def __getitem__(self, idx: int) -> N3dDipole:
        return self.data[idx]

    def __iter__(self) -> N3dDipole:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)
