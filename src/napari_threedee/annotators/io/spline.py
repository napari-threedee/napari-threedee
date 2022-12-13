import napari
from napari.types import LayerDataTuple
import numpy as np
import zarr

from .constants import ANNOTATION_TYPE_KEY, N3D_METADATA_KEY
from ..spline_annotator import SplineAnnotator


def validate_spline_layer(layer: napari.layers.Points):
    """Ensure a spline layer matches the spline layer specification."""
    if SplineAnnotator.SPLINE_ID_FEATURES_KEY not in layer.features:
        error = f"{SplineAnnotator.SPLINE_ID_FEATURES_KEY} not in layer features."
        raise ValueError(error)
    elif SplineAnnotator.SPLINES_KEY not in layer.metadata[N3D_METADATA_KEY]:
        error = f"{SplineAnnotator.SPLINES_KEY} not in n3d metadata entry."
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
    elif SplineAnnotator.SPLINES_KEY not in n3d_zarr.attrs:
        raise ValueError(f"{SplineAnnotator.SPLINES_KEY} not found")


def layer_to_n3d_zarr(layer: napari.layers.Points) -> zarr.Array:
    """Convert a napari points layer into an n3d zarr array."""
    validate_spline_layer(layer)
    n3d_zarr = zarr.array(layer.data)
    n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = SplineAnnotator.ANNOTATION_TYPE

    # get spline id per point from layer features
    spline_id_key = SplineAnnotator.SPLINE_ID_FEATURES_KEY
    n3d_zarr.attrs[spline_id_key] = list(layer.features[spline_id_key])

    # get dict of spline t,c,k tuples from n3d metadata dict
    n3d_metadata = layer.metadata[N3D_METADATA_KEY]
    n3d_zarr.attrs[SplineAnnotator.SPLINES_KEY] = \
        n3d_metadata[SplineAnnotator.SPLINES_KEY]
    return n3d_zarr


def n3d_zarr_to_layer_data_tuple(n3d_zarr: zarr.Array) -> LayerDataTuple:
    """Convert an n3d zarr array to an n3d spline points layer data tuple."""
    validate_spline_zarr(n3d_zarr)
    spline_id = n3d_zarr.attrs[SplineAnnotator.SPLINE_ID_FEATURES_KEY]
    spline_order = n3d_zarr.attrs[SplineAnnotator.SPLINES_KEY]
    layer_kwargs = {
        "features": {SplineAnnotator.SPLINE_ID_FEATURES_KEY: spline_id},
        "metadata": {
            N3D_METADATA_KEY: {SplineAnnotator.SPLINES_KEY: spline_order},
        }
    }
    return (np.array(n3d_zarr), layer_kwargs, "points")
