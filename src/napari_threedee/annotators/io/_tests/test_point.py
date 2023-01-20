import napari.layers
import numpy as np
import pytest
import zarr
from napari.layers import Points

from napari_threedee.annotators.io.point import layer_to_n3d_zarr, \
    n3d_zarr_to_layer_data_tuple, validate_zarr, validate_layer


def test_layer_to_zarr_from_layer_matching_spec(
    valid_n3d_point_layer, tmp_path
):
    """Assert successful if layer matches spec."""
    layer = valid_n3d_point_layer
    validate_layer(layer)
    n3d_zarr = layer_to_n3d_zarr(layer, tmp_path)
    validate_zarr(n3d_zarr)
    assert isinstance(n3d_zarr, zarr.Array)


def test_layer_to_zarr_from_incompatible_layer():
    """Assert non-succesful if zarr doesn't match spec."""
    layer = Points(data=np.random.normal(size=(2, 3)))
    with pytest.raises(ValueError):
        validate_layer(layer)
        layer_to_n3d_zarr(layer)


def test_zarr_to_layer_data_tuple_from_compatible_zarr(valid_n3d_point_zarr):
    """Assert succesful if zarr matches spec."""
    n3d_zarr = valid_n3d_point_zarr
    validate_zarr(n3d_zarr)
    layer_data_tuple = n3d_zarr_to_layer_data_tuple(n3d_zarr)
    points = napari.layers.Layer.create(*layer_data_tuple)
    validate_layer(points)


def test_zarr_to_layer_data_tuple_from_incompatible_zarr():
    """Assert unsuccesful if zarr doesn't match spec."""
    n3d_zarr = zarr.array(np.random.normal(size=(2, 3)))
    with pytest.raises(ValueError):
        validate_zarr(n3d_zarr)
        n3d_zarr_to_layer_data_tuple(n3d_zarr)
