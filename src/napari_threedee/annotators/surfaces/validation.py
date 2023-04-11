import napari
import zarr

from napari_threedee.annotators.surfaces.constants import (
    N3D_METADATA_KEY,
    ANNOTATION_TYPE_KEY,
    SURFACE_ANNOTATION_TYPE_KEY,
    SURFACE_ID_FEATURES_KEY,
    LEVEL_ID_FEATURES_KEY,
)


def validate_layer(layer: napari.layers.Points):
    """Ensure a points layer matches the n3d surface layer specification."""
    if N3D_METADATA_KEY not in layer.metadata:
        raise ValueError(f"{N3D_METADATA_KEY} not in layer metadata.")
    n3d_metadata = layer.metadata[N3D_METADATA_KEY]
    if n3d_metadata[ANNOTATION_TYPE_KEY] != SURFACE_ANNOTATION_TYPE_KEY:
        error = f"Annotation type is not {SURFACE_ANNOTATION_TYPE_KEY}"
        raise ValueError(error)
    for key in (SURFACE_ID_FEATURES_KEY, LEVEL_ID_FEATURES_KEY):
        if key not in layer.features:
            error = f"{key} not in layer features."
            raise ValueError(error)


def validate_n3d_zarr(n3d_zarr: zarr.Array):
    """Ensure an n3d zarr array can be converted to a n3d surface points layer."""
    if ANNOTATION_TYPE_KEY not in n3d_zarr.attrs:
        raise ValueError("cannot read as n3d surfaces.")
    if n3d_zarr.attrs[ANNOTATION_TYPE_KEY] != SURFACE_ANNOTATION_TYPE_KEY:
        raise ValueError("cannot read as n3d surfaces.")
    for key in (SURFACE_ID_FEATURES_KEY, LEVEL_ID_FEATURES_KEY):
        if key not in n3d_zarr.attrs:
            raise ValueError(f"{key} not found.")
