import os

import napari
import numpy as np
import zarr
from napari.types import LayerDataTuple

from napari_threedee.annotators import PointAnnotator
from napari_threedee.annotators.io import ANNOTATION_TYPE_KEY, N3D_METADATA_KEY


def validate_layer(layer: napari.layers.N3dPoints):
    """Ensure a sphere layer matches the specification."""
    if N3D_METADATA_KEY not in layer.metadata:
        raise ValueError(f"{N3D_METADATA_KEY} not found")
    n3d_metadata = layer.metadata[N3D_METADATA_KEY]
    if n3d_metadata[ANNOTATION_TYPE_KEY] != PointAnnotator.ANNOTATION_TYPE:
        raise ValueError("Cannot read as n3d points layer.")


def validate_zarr(n3d_zarr: zarr.Array):
    """Ensure an n3d zarr array contains data for n3d points layer."""
    if ANNOTATION_TYPE_KEY not in n3d_zarr.attrs:
        raise ValueError("cannot read as n3d sphere.")
    if n3d_zarr.attrs[ANNOTATION_TYPE_KEY] != PointAnnotator.ANNOTATION_TYPE:
        raise ValueError("cannot read as n3d sphere.")


def layer_to_n3d_zarr(layer: napari.layers.N3dPoints, path: os.PathLike) -> zarr.Array:
    """Convert an n3d sphere points layer into an n3d zarr array."""
    validate_layer(layer)
    n3d_zarr = zarr.open_array(
        store=path,
        shape=layer.data.shape,
        dtype=layer.data.dtype,
        mode="w"
    )
    n3d_zarr[...] = layer.data
    n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = PointAnnotator.ANNOTATION_TYPE
    return n3d_zarr


def n3d_zarr_to_layer_data_tuple(n3d_zarr: zarr.Array) -> LayerDataTuple:
    """Convert an n3d zarr array into an n3d sphere points layer data tuple."""
    validate_zarr(n3d_zarr)
    layer_kwargs = {
        "metadata": {N3D_METADATA_KEY: {
            ANNOTATION_TYPE_KEY: PointAnnotator.ANNOTATION_TYPE,
        }},
    }
    return (np.array(n3d_zarr), layer_kwargs, "points")