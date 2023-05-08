import napari_threedee as n3d
import numpy as np
import zarr

from napari_threedee.data_models import N3dPoints
from napari_threedee.annotators.points.validation import validate_layer, \
    validate_n3d_zarr


def test_empty_points():
    points = n3d.data_models.N3dPoints(data=[])
    assert points.data.shape == (0, 3)


def test_instantiation():
    points = N3dPoints(data=np.random.uniform(low=0, high=10, size=(10, 3)))
    assert isinstance(points, N3dPoints)


def test_to_and_from_layer():
    points = N3dPoints(data=np.random.uniform(low=0, high=10, size=(10, 3)))
    layer = points.as_layer()
    validate_layer(layer)
    points = N3dPoints.from_layer(layer)
    assert isinstance(points, N3dPoints)


def test_to_and_from_n3d_zarr(tmp_path):
    points = N3dPoints(data=np.random.uniform(low=0, high=10, size=(10, 3)))
    points.to_n3d_zarr(tmp_path)
    n3d_zarr = zarr.open(tmp_path)
    validate_n3d_zarr(n3d_zarr)
    points = N3dPoints.from_n3d_zarr(tmp_path)
    assert isinstance(points, N3dPoints)


def test_empty_points_as_layer():
    points = N3dPoints(data=[])
    layer = points.as_layer()
    assert layer.ndim == 3
