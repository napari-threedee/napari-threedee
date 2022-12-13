import napari.layers
import numpy as np
import pytest
import zarr
from napari.layers import Points

from napari_threedee.annotators.io.sphere import layer_to_n3d_zarr, \
    n3d_zarr_to_layer_data_tuple, validate_sphere_zarr, validate_sphere_layer
from napari_threedee.annotators import SphereAnnotator
from napari_threedee.annotators.io import ANNOTATION_TYPE_KEY, N3D_METADATA_KEY


def test_layer_to_zarr_from_layer_matching_spec():
    """Assert successful if layer matches spec."""
    layer = Points(
        data=np.random.normal(size=(2, 3)),
        features={
            SphereAnnotator.SPHERE_ID_FEATURES_KEY: [0, 1],
            SphereAnnotator.SPHERE_RADIUS_FEATURES_KEY: [5, 5]
        },
        metadata={N3D_METADATA_KEY: dict()}
    )
    validate_sphere_layer(layer)
    n3d_zarr = layer_to_n3d_zarr(layer)
    validate_sphere_zarr(n3d_zarr)
    assert isinstance(n3d_zarr, zarr.Array)


def test_layer_to_zarr_from_incompatible_layer():
    """Assert non-succesful if zarr doesn't match spec."""
    layer = Points(data=np.random.normal(size=(2, 3)))
    with pytest.raises(ValueError):
        validate_sphere_layer(layer)
        layer_to_n3d_zarr(layer)


def test_zarr_to_layer_data_tuple_from_compatible_zarr():
    """Assert succesful if zarr matches spec."""
    n3d_zarr = zarr.array(np.random.normal(size=(2, 3)))
    n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = SphereAnnotator.ANNOTATION_TYPE
    n3d_zarr.attrs[SphereAnnotator.SPHERE_ID_FEATURES_KEY] = [0, 1]
    n3d_zarr.attrs[SphereAnnotator.SPHERE_RADIUS_FEATURES_KEY] = [5, 5]
    validate_sphere_zarr(n3d_zarr)
    layer_data_tuple = n3d_zarr_to_layer_data_tuple(n3d_zarr)
    points = napari.layers.Layer.create(*layer_data_tuple)
    validate_sphere_layer(points)


def test_zarr_to_layer_data_tuple_from_incompatible_zarr():
    """Assert unsuccesful if zarr doesn't match spec."""
    n3d_zarr = zarr.array(np.random.normal(size=(2, 3)))
    with pytest.raises(ValueError):
        validate_sphere_zarr(n3d_zarr)
        n3d_zarr_to_layer_data_tuple(n3d_zarr)
