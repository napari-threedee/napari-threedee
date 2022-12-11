import napari
from napari.types import LayerDataTuple
import numpy as np
import zarr

from .constants import ANNOTATION_TYPE_KEY
from ..spline_annotator import SplineAnnotator


def validate_spline_layer(layer: napari.layers.Points):
    """Ensure a spline layer matches the spline layer specification."""
    if SplineAnnotator.SPLINE_ID_FEATURES_KEY not in layer.features:
        error = f"{SplineAnnotator.SPLINE_ID_FEATURES_KEY} not in layer features."
        raise ValueError(error)
    elif SplineAnnotator.SPLINE_ORDER_KEY not in layer.metadata:
        error = f"{SplineAnnotator.SPLINE_ORDER_KEY} not in layer metadata."
        raise ValueError(error)


def validate_spline_zarr(n3d_zarr: zarr.Array):
    """Ensure an n3d zarr array can be converted to a napari points layer."""
    if ANNOTATION_TYPE_KEY not in n3d_zarr.attrs:
        raise ValueError("cannot read as n3d spline zarr.")
    elif n3d_zarr.attrs["annotation_type"] != SplineAnnotator.ANNOTATION_TYPE:
        raise ValueError("cannot read as n3d spline zarr.")
    elif SplineAnnotator.SPLINE_ID_FEATURES_KEY not in n3d_zarr.attrs:
        raise ValueError(f"{SplineAnnotator.SPLINE_ID_FEATURES_KEY} not found.")
    elif SplineAnnotator.SPLINE_ORDER_KEY not in n3d_zarr.attrs:
        raise ValueError(f"{SplineAnnotator.SPLINE_ORDER_KEY} not found")


def layer_to_zarr(layer: napari.layers.Points) -> zarr.Array:
    """Convert a napari points layer into an n3d zarr array."""
    validate_spline_layer(layer)
    n3d = zarr.array(layer.data)
    n3d.attrs[ANNOTATION_TYPE_KEY] = SplineAnnotator.ANNOTATION_TYPE
    spline_id_key = SplineAnnotator.SPLINE_ID_FEATURES_KEY
    n3d.attrs[spline_id_key] = list(layer.features[spline_id_key])
    spline_order_key = SplineAnnotator.SPLINE_ORDER_KEY
    n3d.attrs[spline_order_key] = layer.metadata[spline_order_key]
    return n3d


def zarr_to_layer_data_tuple(n3d_zarr: zarr.Array) -> LayerDataTuple:
    """Convert an n3d zarr array to a napari points layer."""
    validate_spline_zarr(n3d_zarr)
    spline_id = n3d_zarr.attrs[SplineAnnotator.SPLINE_ID_FEATURES_KEY]
    spline_order = n3d_zarr.attrs[SplineAnnotator.SPLINE_ORDER_KEY]
    layer_kwargs = {
        "features": {SplineAnnotator.SPLINE_ID_FEATURES_KEY: spline_id},
        "metadata": {SplineAnnotator.SPLINE_ORDER_KEY: spline_order},
    }
    return (np.array(n3d_zarr), layer_kwargs, "points")
