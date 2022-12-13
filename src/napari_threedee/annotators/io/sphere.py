import napari
import numpy as np
import zarr
from napari.types import LayerDataTuple

from napari_threedee.annotators import SphereAnnotator
from napari_threedee.annotators.io import ANNOTATION_TYPE_KEY


def validate_sphere_layer(layer: napari.layers.Points):
    """Ensure a sphere layer matches the specification."""
    for feature in (SphereAnnotator.SPHERE_ID_KEY,
                    SphereAnnotator.SPHERE_RADIUS_KEY):
        if feature not in layer.features:
            raise ValueError(f"{feature} not in layer features.")


def validate_sphere_zarr(n3d_zarr: zarr.Array):
    """Ensure an n3d zarr array contains data for n3d sphere points layer."""
    if ANNOTATION_TYPE_KEY not in n3d_zarr.attrs:
        raise ValueError("cannot read as n3d sphere.")
    if n3d_zarr.attrs[ANNOTATION_TYPE_KEY] != SphereAnnotator.ANNOTATION_TYPE:
        raise ValueError("cannot read as n3d sphere.")


def layer_to_n3d_zarr(layer: napari.layers.Points) -> zarr.Array:
    """Convert an n3d sphere points layer into an n3d zarr array."""
    validate_sphere_layer(layer)
    n3d_zarr = zarr.array(layer.data)
    n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = SphereAnnotator.ANNOTATION_TYPE

    # get sphere id and radius from layer features
    id_key = SphereAnnotator.SPHERE_ID_KEY
    n3d_zarr.attrs[id_key] = list(layer.features[id_key])
    radius_key = SphereAnnotator.SPHERE_RADIUS_KEY
    n3d_zarr.attrs[radius_key] = list(layer.features[radius_key])
    return n3d_zarr


def n3d_zarr_to_layer_data_tuple(n3d_zarr: zarr.Array) -> LayerDataTuple:
    """Convert an n3d zarr array into an n3d sphere points layer data tuple."""
    validate_sphere_zarr(n3d_zarr)
    id = n3d_zarr.attrs[SphereAnnotator.SPHERE_ID_KEY]
    radii = n3d_zarr.attrs[SphereAnnotator.SPHERE_RADIUS_KEY]
    layer_kwargs = {
        "features": {
            SphereAnnotator.SPHERE_ID_KEY: id,
            SphereAnnotator.SPHERE_RADIUS_KEY: radii,
        }
    }
    return (np.array(n3d_zarr), layer_kwargs, "points")