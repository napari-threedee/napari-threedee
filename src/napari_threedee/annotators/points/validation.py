import napari.layers
import zarr

from napari_threedee.annotators.constants import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY
from napari_threedee.annotators.points.constants import POINT_ANNOTATION_TYPE_KEY


def validate_layer(layer: napari.layers.Points):
    """Ensure a napari points layer matches the n3d layer specification."""
    if N3D_METADATA_KEY not in layer.metadata:
        raise ValueError(f"{N3D_METADATA_KEY} not found")
    n3d_metadata = layer.metadata[N3D_METADATA_KEY]
    if n3d_metadata[ANNOTATION_TYPE_KEY] != POINT_ANNOTATION_TYPE_KEY:
        raise ValueError("Cannot read as n3d points layer.")


def validate_n3d_zarr(n3d_zarr: zarr.Array):
    """Ensure an n3d zarr array contains data for n3d points layer."""
    if ANNOTATION_TYPE_KEY not in n3d_zarr.attrs:
        raise ValueError("cannot read as n3d points.")
    if n3d_zarr.attrs[ANNOTATION_TYPE_KEY] != POINT_ANNOTATION_TYPE_KEY:
        raise ValueError("cannot read as n3d points.")
