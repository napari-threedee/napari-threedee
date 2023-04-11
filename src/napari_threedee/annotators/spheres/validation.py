import napari
import zarr

from napari_threedee.annotators.constants import ANNOTATION_TYPE_KEY
from napari_threedee.annotators.spheres.constants import SPHERE_ANNOTATION_TYPE_KEY, \
    SPHERE_ID_FEATURES_KEY, SPHERE_RADIUS_FEATURES_KEY


def validate_layer(layer: napari.layers.Points):
    """Ensure a sphere layer matches the specification."""
    for feature in (SPHERE_ID_FEATURES_KEY, SPHERE_RADIUS_FEATURES_KEY):
        if feature not in layer.features:
            raise ValueError(f"{feature} not in layer features.")


def validate_n3d_zarr(n3d_zarr: zarr.Array):
    """Ensure an n3d zarr array contains data for n3d sphere points layer."""
    if ANNOTATION_TYPE_KEY not in n3d_zarr.attrs:
        raise ValueError("cannot read as n3d sphere.")
    if n3d_zarr.attrs[ANNOTATION_TYPE_KEY] != SPHERE_ANNOTATION_TYPE_KEY:
        raise ValueError("cannot read as n3d sphere.")
