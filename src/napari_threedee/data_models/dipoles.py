import os
from typing import List, Tuple

import napari
import numpy as np
import zarr
from pydantic import BaseModel, validator

from napari_threedee.annotators.base import N3dDataModel
from napari_threedee.annotators.constants import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY
from napari_threedee.data_models.spline_sampler import SplineSampler
from napari_threedee.annotators.paths.validation import (
    validate_layer,
    validate_n3d_zarr,
)
from napari_threedee.annotators.paths.constants import (
    PATH_ID_FEATURES_KEY,
    PATH_ANNOTATION_TYPE_KEY,
    COLOR_CYCLE,
)


class N3dDipole(BaseModel):
    center: Tuple[float, float, float]
    direction: Tuple[float, float, float]


class N3dDipoles(N3dDataModel):
    data: List[N3dDipole]

    # @classmethod
    # def from_layer(cls, layer: napari.layers.Layer):
    #    grouped_points_features = layer.features.groupby(PATH_ID_FEATURES_KEY)
    #    splines = [
    #        N3dPath(data=layer.data[df.index.tolist()])
    #        for name, df in grouped_points_features
    #    ]
    #    return cls(data=splines)

    def as_layer(self) -> napari.layers.Points:

        #if len(self) == 0:  # workaround for napari/napari#4213
        #    cls = type(self)
        #    n3d_paths = cls(data=[N3dPath(data=[0, 0, 0])])
        #    layer = n3d_paths.as_layer()
        #    layer.selected_data = {0}
        #    layer.remove_selected()
        #    return layer

        # Construct (n,3) np array of center points
        centers = np.stack([dipole.center for dipole in self.data], axis=0)

        # Construct the (n,3) np array for directions
        directions = np.stack([dipole.direction for dipole in self.data], axis=0)

        # Construct the points layer from the data
        # ... soon


'''
        data = np.concatenate([spline.data for spline in self.data])
        metadata = {
            N3D_METADATA_KEY: {
                ANNOTATION_TYPE_KEY: PATH_ANNOTATION_TYPE_KEY,
            }
        }
        features = {PATH_ID_FEATURES_KEY: self.path_ids}
        layer = napari.layers.Points(
            data=data,
            metadata=metadata,
            features=features,
            face_color=PATH_ID_FEATURES_KEY,
            face_color_cycle=COLOR_CYCLE,
            name='n3d paths',
            ndim=data.shape[-1],
        )
        layer.selected_data = {len(data) - 1}
        validate_layer(layer)
        return layer

    @classmethod
    def from_n3d_zarr(cls, path: os.PathLike):
        n3d_zarr = zarr.open(path)
        validate_n3d_zarr(n3d_zarr)
        points = np.asarray(n3d_zarr)
        spline_ids = n3d_zarr.attrs[PATH_ID_FEATURES_KEY]
        splines = [
            N3dPath(data=points[spline_ids == idx])
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
        n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = PATH_ANNOTATION_TYPE_KEY
        n3d_zarr.attrs[PATH_ID_FEATURES_KEY] = list(self.path_ids)

    @classmethod
    def create_empty_layer(cls, ndim: int):
        """Create an empty path annotator points layer.

        Parameters
        ----------
        ndim : int
            The dimensionality of the empty points layer.
            Generally, this should match the image layer being
            annotated.

        Returns
        -------
        layer : napari.layers.Points
            The napari Points layer initialized for path annotation.
        """
        # workaround for napari/napari#4213
        dummy_data = np.zeros((ndim,))
        n3d_paths = cls(data=[N3dPath(data=dummy_data)])
        layer = n3d_paths.as_layer()
        layer.selected_data = {0}
        layer.remove_selected()
        return layer

    def __getitem__(self, idx: int) -> N3dPath:
        return self.data[idx]

    def __iter__(self) -> N3dPath:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)
'''