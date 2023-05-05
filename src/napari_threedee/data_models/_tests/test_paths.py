import napari
import numpy as np
import zarr

from napari_threedee.data_models.paths import N3dPath, N3dPaths
from napari_threedee.annotators.paths.validation import validate_layer, \
    validate_n3d_zarr


def test_single_path_instantiation():
    path = N3dPath(data=np.random.uniform(0, 10, size=(10, 3)))
    assert isinstance(path, N3dPath)


def test_single_path_interpolation():
    path = N3dPath(data=np.random.uniform(0, 10, size=(10, 3)))
    result = path.sample(n=10000)
    assert result.shape == (10000, 3)


def test_paths_instantiation():
    path = N3dPath(data=np.random.uniform(0, 10, size=(10, 3)))
    paths = N3dPaths(data=[path, path])
    assert isinstance(paths, N3dPaths)
    assert paths.n_points == 20
    assert paths.ndim == 3


def test_paths_to_and_from_layer():
    path = N3dPath(data=np.random.uniform(0, 10, size=(10, 3)))
    paths = N3dPaths(data=[path, path])
    layer = paths.as_layer()
    validate_layer(layer)
    paths = N3dPaths.from_layer(layer)
    assert isinstance(paths, N3dPaths)


def test_paths_to_and_from_n3d_zarr(tmp_path):
    path = N3dPath(data=np.random.uniform(0, 10, size=(10, 3)))
    paths = N3dPaths(data=[path, path])
    paths.to_n3d_zarr(tmp_path)
    n3d_zarr = zarr.open(tmp_path)
    validate_n3d_zarr(n3d_zarr)
    paths = N3dPaths.from_n3d_zarr(tmp_path)
    assert isinstance(paths, N3dPaths)


def test_empty_path():
    path = N3dPath(data=[])
    assert path.data.shape == (0, 3)


def test_empty_paths_as_layer():
    paths = N3dPaths(data=[])
    layer = paths.as_layer()
    assert isinstance(layer, napari.layers.Points)
