import napari.layers
import numpy as np
import pytest
import zarr
from napari.layers import Points

from napari_threedee.annotators.io.spline import layer_to_zarr, \
    zarr_to_layer_data_tuple, validate_spline_zarr, validate_spline_layer
from napari_threedee.annotators import SplineAnnotator
from napari_threedee.annotators.io import ANNOTATION_TYPE_KEY, N3D_METADATA_KEY


def test_layer_to_zarr_from_layer_matching_spec():
    """Assert successful if layer matches spec."""
    layer = Points(
        data=np.random.normal(size=(2, 3)),
        features={SplineAnnotator.SPLINE_ID_FEATURES_KEY: [0, 1]},
        metadata={N3D_METADATA_KEY: {SplineAnnotator.SPLINES_KEY: {}}}
    )
    validate_spline_layer(layer)
    n3d_zarr = layer_to_zarr(layer)
    validate_spline_zarr(n3d_zarr)
    assert isinstance(n3d_zarr, zarr.Array)


def test_layer_to_zarr_from_incompatible_layer():
    """Assert non-succesful if zarr doesn't match spec."""
    layer = Points(data=np.random.normal(size=(2, 3)))
    with pytest.raises(ValueError):
        validate_spline_layer(layer)
        layer_to_zarr(layer)


def test_zarr_to_layer_data_tuple_from_compatible_zarr():
    """Assert succesful if zarr matches spec."""
    n3d_zarr = zarr.array(np.random.normal(size=(2, 3)))
    n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = SplineAnnotator.ANNOTATION_TYPE
    n3d_zarr.attrs[SplineAnnotator.SPLINE_ID_FEATURES_KEY] = [0, 1]
    n3d_zarr.attrs[SplineAnnotator.SPLINES_KEY] = {}
    validate_spline_zarr(n3d_zarr)
    layer_data_tuple = zarr_to_layer_data_tuple(n3d_zarr)
    points = napari.layers.Layer.create(*layer_data_tuple)
    validate_spline_layer(points)


def test_zarr_to_layer_data_tuple_from_incompatible_zarr():
    """Assert unsuccesful if zarr doesn't match spec."""
    n3d_zarr = zarr.array(np.random.normal(size=(2, 3)))
    with pytest.raises(ValueError):
        validate_spline_zarr(n3d_zarr)
        zarr_to_layer_data_tuple(n3d_zarr)
