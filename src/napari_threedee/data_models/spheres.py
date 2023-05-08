import os
from typing import Tuple

import napari
import numpy as np
import zarr
from pydantic import validator

from napari_threedee.annotators.base import N3dDataModel
from napari_threedee.annotators.spheres.validation import (
    validate_layer,
    validate_n3d_zarr,
)
from napari_threedee.annotators.spheres.constants import (
    N3D_METADATA_KEY,
    ANNOTATION_TYPE_KEY,
    SPHERE_ANNOTATION_TYPE_KEY,
    SPHERE_RADIUS_FEATURES_KEY,
    SPHERE_ID_FEATURES_KEY,
    COLOR_CYCLE,
)


class N3dSpheres(N3dDataModel):
    centers: np.ndarray
    radii: np.ndarray

    @classmethod
    def from_layer(cls, layer: napari.layers.Points):
        centers = np.array(layer.data, dtype=np.float32)
        radii = np.array(layer.features[SPHERE_RADIUS_FEATURES_KEY], dtype=np.float32)
        return cls(centers=centers, radii=radii)

    def as_layer(self) -> napari.layers.Points:
        if len(self.centers) == 0:  # workaround for napari/napari#4213
            cls = type(self)
            layer = cls(centers=[0, 0, 0], radii=[10]).as_layer()
            layer.selected_data = {0}
            layer.remove_selected()
            return layer
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

    @validator('centers', pre=True)
    def ensure_at_least_2d(cls, value):
        data = np.atleast_2d(value)
        if data.shape[-1] == 0:
            data = np.zeros(shape=(0, 3))
        return data

    def to_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        from vispy.geometry import create_sphere
        sphere_vertices = []
        sphere_faces = []
        face_index_offset = 0
        for idx, (center, radius) in enumerate(zip(self.centers, self.radii)):
            mesh_data = create_sphere(radius=radius, rows=20, cols=20)
            vertex_data = mesh_data.get_vertices() + center
            sphere_vertices.append(vertex_data)
            sphere_faces.append(mesh_data.get_faces() + face_index_offset)
            face_index_offset += len(vertex_data)
        if len(sphere_vertices) > 0:
            sphere_vertices = np.concatenate(sphere_vertices, axis=0)
            sphere_faces = np.concatenate(sphere_faces, axis=0)
        else:
            sphere_vertices, sphere_faces = np.array([]), np.array([])
        return sphere_vertices, sphere_faces
