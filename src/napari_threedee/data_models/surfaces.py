import os
from typing import List

import napari
import numpy as np
import pandas as pd
import zarr
from pydantic import BaseModel, validator

from napari_threedee.annotators.base import N3dDataModel
from napari_threedee.annotators.constants import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY
from napari_threedee.annotators.surfaces.constants import LEVEL_ID_FEATURES_KEY, \
    SURFACE_ID_FEATURES_KEY, SURFACE_ANNOTATION_TYPE_KEY, COLOR_CYCLE
from napari_threedee.annotators.surfaces.validation import validate_layer, \
    validate_n3d_zarr


class N3dSurface(BaseModel):
    data: List[np.ndarray]

    class Config:
        arbitrary_types_allowed = True

    @property
    def ndim(self) -> int:
        return self.data[0].shape[-1]

    @validator('data', pre=True)
    def ensure_float32_ndarray_list(cls, value):
        data = [
            np.asarray(arr, dtype=np.float32)
            for arr
            in value
        ]
        return data

    @property
    def n_points(self) -> int:
        return sum(len(arr) for arr in self.data)

    @property
    def n_levels(self) -> int:
        return len(self.data)

    @property
    def points(self) -> np.ndarray:
        return np.concatenate([level for level in self.data])

    @property
    def level_ids(self) -> np.ndarray:
        level_ids = [
            [idx] * len(level)
            for idx, level
            in enumerate(self.data)
        ]
        return np.concatenate(level_ids)

    def __iter__(self) -> np.ndarray:
        yield from self.data


class N3dSurfaces(N3dDataModel):
    data: List[N3dSurface]

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def ndim(self) -> int:
        return self.data[0].ndim

    @property
    def points(self) -> np.ndarray:
        return np.concatenate([surface.points for surface in self.data])

    @property
    def surface_ids(self) -> np.ndarray:
        surface_ids = [
            [idx] * surface.n_points
            for idx, surface
            in enumerate(self.data)
        ]
        return np.concatenate(surface_ids)

    @property
    def level_ids(self) -> np.ndarray:
        level_ids = np.concatenate(
            [
                surface.level_ids
                for surface
                in self.data
            ]
        )
        return level_ids

    @classmethod
    def from_layer(cls, layer: napari.layers.Layer):
        surfaces = [
            N3dSurface(
                data=[
                    layer.data[level_df.index.tolist()]
                    for _, level_df
                    in surface_df.groupby(LEVEL_ID_FEATURES_KEY)
                ]
            )
            for _, surface_df
            in layer.features.groupby(SURFACE_ID_FEATURES_KEY)
        ]
        return cls(data=surfaces)

    def as_layer(self) -> napari.layers.Points:
        metadata = {
            N3D_METADATA_KEY: {
                ANNOTATION_TYPE_KEY: SURFACE_ANNOTATION_TYPE_KEY,
            }
        }
        features = {
            SURFACE_ID_FEATURES_KEY: self.surface_ids,
            LEVEL_ID_FEATURES_KEY: self.level_ids,
        }
        layer = napari.layers.Points(
            data=self.points,
            metadata=metadata,
            features=features,
            face_color=SURFACE_ID_FEATURES_KEY,
            face_color_cycle=COLOR_CYCLE,
            name='n3d surfaces',
            ndim=self.points.shape[-1],
        )
        validate_layer(layer)
        return layer

    @classmethod
    def from_n3d_zarr(cls, path: os.PathLike):
        n3d_zarr = zarr.open(path)
        validate_n3d_zarr(n3d_zarr)
        points = np.asarray(n3d_zarr)
        surface_ids = n3d_zarr.attrs[SURFACE_ID_FEATURES_KEY]
        level_ids = n3d_zarr.attrs[LEVEL_ID_FEATURES_KEY]
        df = pd.DataFrame(
            {
                SURFACE_ID_FEATURES_KEY: surface_ids,
                LEVEL_ID_FEATURES_KEY: level_ids,
            }
        )
        surfaces = [
            N3dSurface(
                data=[
                    points[level_df.index.tolist()]
                    for _, level_df
                    in surface_df.groupby(LEVEL_ID_FEATURES_KEY)
                ]
            )
            for _, surface_df
            in df.groupby(SURFACE_ID_FEATURES_KEY)
        ]
        return cls(data=surfaces)

    def to_n3d_zarr(self, path: os.PathLike) -> None:
        n3d_zarr = zarr.open_array(
            store=path,
            shape=(self.n_points, self.ndim),
            dtype=np.float32,
            mode="w",
        )
        n3d_zarr[...] = self.points
        n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = SURFACE_ANNOTATION_TYPE_KEY
        n3d_zarr.attrs[SURFACE_ID_FEATURES_KEY] = list(self.surface_ids)
        n3d_zarr.attrs[LEVEL_ID_FEATURES_KEY] = list(self.level_ids)

    def __getitem__(self, idx: int) -> N3dSurface:
        return self.data[idx]

    def __iter__(self) -> N3dSurface:
        yield from self.data