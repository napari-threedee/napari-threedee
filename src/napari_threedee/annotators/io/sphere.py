import os

import napari
import numpy as np
import zarr
from napari.types import LayerDataTuple

from napari_threedee.annotators import SphereAnnotator
from napari_threedee.annotators.io import ANNOTATION_TYPE_KEY, N3D_METADATA_KEY


def layer_to_n3d_zarr(layer: napari.layers.Points, path: os.PathLike) -> zarr.Array:
    """Convert an n3d sphere points layer into an n3d zarr array."""
    validate_layer(layer)
    n3d_zarr = zarr.open_array(
        store=path,
        shape=layer.data.shape,
        dtype=layer.data.dtype,
        mode="w"
    )
    n3d_zarr[...] = layer.data
    n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = SphereAnnotator.ANNOTATION_TYPE

    # get sphere id and radius from layer features
    id_key = SphereAnnotator.SPHERE_ID_FEATURES_KEY
    n3d_zarr.attrs[id_key] = list(layer.features[id_key])
    radius_key = SphereAnnotator.SPHERE_RADIUS_FEATURES_KEY
    n3d_zarr.attrs[radius_key] = list(layer.features[radius_key])
    return n3d_zarr


def n3d_zarr_to_layer_data_tuple(n3d_zarr: zarr.Array) -> LayerDataTuple:
    """Convert an n3d zarr array into an n3d sphere points layer data tuple."""
    validate_zarr(n3d_zarr)
    id = n3d_zarr.attrs[SphereAnnotator.SPHERE_ID_FEATURES_KEY]
    radii = n3d_zarr.attrs[SphereAnnotator.SPHERE_RADIUS_FEATURES_KEY]
    layer_kwargs = {
        "features": {
            SphereAnnotator.SPHERE_ID_FEATURES_KEY: id,
            SphereAnnotator.SPHERE_RADIUS_FEATURES_KEY: radii,
        },
        "metadata": {N3D_METADATA_KEY: {
            ANNOTATION_TYPE_KEY: SphereAnnotator.ANNOTATION_TYPE,
        }},
    }
    return (np.array(n3d_zarr), layer_kwargs, "points")