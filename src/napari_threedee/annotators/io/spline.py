import os

import napari
from napari.types import LayerDataTuple
import numpy as np
import zarr

from .constants import ANNOTATION_TYPE_KEY, N3D_METADATA_KEY
from ..spline_annotator import SplineAnnotator


def validate_spline_layer(layer: napari.layers.Points):
    """Ensure a spline layer matches the spline layer specification."""
    if N3D_METADATA_KEY not in layer.metadata:
        raise ValueError(f"{N3D_METADATA_KEY} not in layer metadata.")
    n3d_metadata = layer.metadata[N3D_METADATA_KEY]
    if n3d_metadata[ANNOTATION_TYPE_KEY] != SplineAnnotator.ANNOTATION_TYPE:
        error = f"Annotation type is not {SplineAnnotator.ANNOTATION_TYPE}"
        raise ValueError(error)
    elif SplineAnnotator.SPLINE_ID_FEATURES_KEY not in layer.features:
        error = f"{SplineAnnotator.SPLINE_ID_FEATURES_KEY} not in layer features."
        raise ValueError(error)


def validate_spline_zarr(n3d_zarr: zarr.Array):
    """Ensure an n3d zarr array can be converted to a n3d spline points
    layer."""
    if ANNOTATION_TYPE_KEY not in n3d_zarr.attrs:
        raise ValueError("cannot read as n3d spline.")
    elif n3d_zarr.attrs[ANNOTATION_TYPE_KEY] != SplineAnnotator.ANNOTATION_TYPE:
        raise ValueError("cannot read as n3d spline.")
    elif SplineAnnotator.SPLINE_ID_FEATURES_KEY not in n3d_zarr.attrs:
        raise ValueError(f"{SplineAnnotator.SPLINE_ID_FEATURES_KEY} not found.")


def layer_to_n3d_zarr(layer: napari.layers.Points,
                      path: os.PathLike) -> zarr.Array:
    """Convert a napari points layer into an n3d zarr array."""
    validate_spline_layer(layer)
    n3d_zarr = zarr.open_array(
        store=path,
        shape=layer.data.shape,
        dtype=layer.data.dtype,
        mode="w"
    )
    n3d_zarr[...] = layer.data
    n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = SplineAnnotator.ANNOTATION_TYPE

    spline_id_key = SplineAnnotator.SPLINE_ID_FEATURES_KEY
    n3d_zarr.attrs[spline_id_key] = list(layer.features[spline_id_key])
    return n3d_zarr


def n3d_zarr_to_layer_data_tuple(n3d_zarr: zarr.Array) -> LayerDataTuple:
    """Convert an n3d zarr array to an n3d spline points layer data tuple."""
    validate_spline_zarr(n3d_zarr)
    spline_id = n3d_zarr.attrs[SplineAnnotator.SPLINE_ID_FEATURES_KEY]
    layer_kwargs = {
        "features": {SplineAnnotator.SPLINE_ID_FEATURES_KEY: spline_id},
        "metadata": {
            N3D_METADATA_KEY: {
                ANNOTATION_TYPE_KEY: SplineAnnotator.ANNOTATION_TYPE,
            }
        }
    }
    return (np.array(n3d_zarr), layer_kwargs, "points")
