import os
from typing import List

import einops
import napari
import numpy as np
import zarr
from pydantic import BaseModel, field_validator

from napari_threedee.annotators.base import N3dDataModel
from napari_threedee.annotators.constants import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY
from napari_threedee.annotators.dipoles.validation import validate_layer, validate_n3d_zarr


class N3dDisk(BaseModel):
    center: np.ndarray
    normal_vector: np.ndarray
    radius: float

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    @field_validator('center', 'direction', mode='before')
    def ensure_float32_ndarray(cls, value):
        return np.asarray(value, dtype=np.float32)


class N3dDisks(N3dDataModel):
    data: List[N3dDisk]

    @property
    def centers(self) -> np.ndarray:
        if len(self.data) == 0:
            return np.empty((0, 3))
        return np.stack([disks.center for disks in self.data], axis=0)

    @property
    def normal_vectors(self) -> np.ndarray:
        if len(self.data) == 0:
            return np.empty((0, 3))
        return np.stack([disk.normal_vector for disk in self.data], axis=0)

    @property
    def radii(self) -> np.ndarray:
        if len(self.data) == 0:
            return np.empty((0))
        return np.stack([disk.radius for disk in self.data], axis=0)

    def as_napari_surface_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate vertices and face indices

        returns: vertices, faces
        """
        # generate points on circle
        n_points = 20
        theta = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
        circle_x = self.radius * np.cos(theta)
        circle_y = self.radius * np.sin(theta)
        circle_z = np.zeros_like(circle_x)
        circle_xyz = np.stack((circle_x, circle_y, circle_z), axis=-1)  # (b1, 3)

        # rotate circles for each disk
        circle_xyz = einops.rearrange(circle_xyz, 'b1 xyz -> b1 1 xyz 1')
        rotation_matrices = _vectors_to_rotation_matrices(self.normal_vectors)  # (b2, 3, 3)
        rotation_matrices = einops.rearrange(rotation_matrices, 'b2 i j -> 1 b2 i j')
        rotated_circle_xyz = rotation_matrices @ circle_xyz  # (b1, b2, 3, 1)
        rotated_circle_xyz = einops.rearrange(rotated_circle_xyz, 'b1 b2 xyz 1 -> (b1 b2) xyz')

        # set up array of vertices
        centers = self.centers
        vertices = np.concatenate([rotated_circle_xyz, centers])

        # get indices for each vertex of every triangle for every disk
        n_triangles = n_points
        v0 = np.arange(len(self.centers)) + len(rotated_circle_xyz)  # one per disk

        v1 = np.arange(n_triangles)  # n_triangles per disk
        v2 = np.roll(v1, shift=-1)  # [1, 2, ..., n-1, 0]  n_triangles per disk
        faces = [
            np.stack(
                [
                    np.array([v0] * n_triangles, dtype=np.float32),
                    v1 * i,
                    v2 * i
                ],
                axis=-1
            )
            for i in range(n_triangles)
        ]
        return vertices, faces

    @classmethod
    def from_layer(cls, layer: napari.layers.Layer):
        centers = np.asarray(layer.data)
        directions = np.stack([layer.features[DIPOLE_DIRECTION_Z_FEATURES_KEY],
                               layer.features[DIPOLE_DIRECTION_Y_FEATURES_KEY],
                               layer.features[DIPOLE_DIRECTION_X_FEATURES_KEY]], axis=-1)
        return cls.from_centers_and_(centers=centers, directions=directions)

    def as_layer(self) -> napari.layers.Points:
        if len(self.data) == 0:
            return N3dDisks.create_empty_layer()

        directions = self.normal_vectors
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
            N3dDisk(center=center, direction=direction)
            for center, direction
            in zip(centers, directions)
        ]
        return cls(data=dipoles)

    def to_n3d_zarr(self, path: os.PathLike) -> None:
        n3d_zarr = zarr.open_array(
            store=path,
            shape=(len(self), 2, 3),  # vectors shape: (n, 2, 3)
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
            DIPOLE_ID_FEATURES_KEY: np.array([0]),
            DIPOLE_DIRECTION_X_FEATURES_KEY: np.array([0]),
            DIPOLE_DIRECTION_Y_FEATURES_KEY: np.array([0]),
            DIPOLE_DIRECTION_Z_FEATURES_KEY: np.array([0]),
        }
        metadata = {N3D_METADATA_KEY: {ANNOTATION_TYPE_KEY: DIPOLE_ANNOTATION_TYPE_KEY}}

        # workaround for napari/napari#4213
        dummy_data = np.array([[0, 0, 0]])
        layer = napari.layers.Points(
            data=dummy_data,
            features=features,
            metadata=metadata,
            name='n3d dipoles',
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


def _vectors_to_rotation_matrices(z_vectors):
    # Normalize z vectors
    z = z_vectors / np.linalg.norm(z_vectors, axis=1, keepdims=True)

    # Create reference vectors (trying [1,0,0] for all points)
    ref = np.array([1.0, 0.0, 0.0])

    # Find dots between ref and z
    dot_products = np.einsum('bz,z', ref, z)

    # Where dot product is too large vectors are aligned, use [0,1,0] instead
    mask = dot_products > 0.9
    ref[mask] = [0.0, 1.0, 0.0]

    # Compute x vectors using cross product
    x = np.cross(ref, z)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    # Compute y vectors
    y = np.cross(z, x)

    # Stack the vectors into rotation matrices
    rotation_matrices = np.stack([x, y, z], axis=2)

    return rotation_matrices
