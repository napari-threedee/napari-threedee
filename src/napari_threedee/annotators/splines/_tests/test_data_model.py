import numpy as np
import zarr

from napari_threedee.annotators.splines.data_model import N3dSplines, _N3dSpline
from napari_threedee.annotators.splines.validation import validate_layer, validate_n3d_zarr


def test_single_spline_instantiation():
    spline = _N3dSpline(data=np.random.uniform(0, 10, size=(10, 3)))
    assert isinstance(spline, _N3dSpline)


def test_single_spline_interpolation():
    spline = _N3dSpline(data=np.random.uniform(0, 10, size=(10, 3)))
    result = spline.interpolate(n=10000)
    assert result.shape == (10000, 3)


def test_splines_instantiation():
    spline = _N3dSpline(data=np.random.uniform(0, 10, size=(10, 3)))
    splines = N3dSplines(data=[spline, spline])
    assert isinstance(splines, N3dSplines)
    assert splines.n_points == 20
    assert splines.ndim == 3


def test_splines_to_and_from_layer():
    spline = _N3dSpline(data=np.random.uniform(0, 10, size=(10, 3)))
    splines = N3dSplines(data=[spline, spline])
    layer = splines.as_layer()
    validate_layer(layer)
    splines = N3dSplines.from_layer(layer)
    assert isinstance(splines, N3dSplines)


def test_splines_to_and_from_n3d_zarr(tmp_path):
    spline = _N3dSpline(data=np.random.uniform(0, 10, size=(10, 3)))
    splines = N3dSplines(data=[spline, spline])
    splines.to_n3d_zarr(tmp_path)
    n3d_zarr = zarr.open(tmp_path)
    validate_n3d_zarr(n3d_zarr)
    splines = N3dSplines.from_n3d_zarr(tmp_path)
    assert isinstance(splines, N3dSplines)