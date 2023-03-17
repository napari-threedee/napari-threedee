import napari
import zarr

from napari_threedee.annotators.splines.constants import N3D_METADATA_KEY, \
    ANNOTATION_TYPE_KEY, SPLINE_ANNOTATION_TYPE_KEY, SPLINE_ID_FEATURES_KEY
from napari_threedee.annotators.splines.annotator import SplineAnnotator


def validate_layer(layer: napari.layers.Points):
    """Ensure a spline layer matches the spline layer specification."""
    if N3D_METADATA_KEY not in layer.metadata:
        raise ValueError(f"{N3D_METADATA_KEY} not in layer metadata.")
    n3d_metadata = layer.metadata[N3D_METADATA_KEY]
    if n3d_metadata[ANNOTATION_TYPE_KEY] != SPLINE_ANNOTATION_TYPE_KEY:
        error = f"Annotation type is not {SPLINE_ANNOTATION_TYPE_KEY}"
        raise ValueError(error)
    elif SPLINE_ID_FEATURES_KEY not in layer.features:
        error = f"{SPLINE_ID_FEATURES_KEY} not in layer features."
        raise ValueError(error)


def validate_n3d_zarr(n3d_zarr: zarr.Array):
    """Ensure an n3d zarr array can be converted to a n3d spline points layer."""
    if ANNOTATION_TYPE_KEY not in n3d_zarr.attrs:
        raise ValueError("cannot read as n3d spline.")
    elif n3d_zarr.attrs[ANNOTATION_TYPE_KEY] != SPLINE_ANNOTATION_TYPE_KEY:
        raise ValueError("cannot read as n3d spline.")
    elif SPLINE_ID_FEATURES_KEY not in n3d_zarr.attrs:
        raise ValueError(f"{SPLINE_ID_FEATURES_KEY} not found.")
