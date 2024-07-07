import napari
import zarr

from napari_threedee.annotators.dipoles.constants import (
    ANNOTATION_TYPE_KEY,
    DIPOLE_ANNOTATION_TYPE_KEY,
    DIPOLE_DIRECTION_X_FEATURES_KEY,
    DIPOLE_DIRECTION_Y_FEATURES_KEY,
    DIPOLE_DIRECTION_Z_FEATURES_KEY,
)


def validate_layer(layer: napari.layers.Points):
    """Ensure a sphere layer matches the specification."""
    dipole_direction_features = (
        DIPOLE_DIRECTION_X_FEATURES_KEY,
        DIPOLE_DIRECTION_Y_FEATURES_KEY,
        DIPOLE_DIRECTION_Z_FEATURES_KEY
    )
    for feature in dipole_direction_features:
        if feature not in layer.features:
            raise ValueError(f"{feature} not in layer features.")


def validate_n3d_zarr(n3d_zarr: zarr.Array):
    """Ensure an n3d zarr array can be converted to a n3d dipoles points layer."""
    if ANNOTATION_TYPE_KEY not in n3d_zarr.attrs:
        raise ValueError("cannot read as n3d dipoles.")
    elif n3d_zarr.attrs[ANNOTATION_TYPE_KEY] != DIPOLE_ANNOTATION_TYPE_KEY:
        raise ValueError("cannot read as n3d dipoles.")

