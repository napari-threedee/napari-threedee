import numpy as np
import zarr

from napari_threedee.data_models import N3dSurface, N3dSurfaces
from napari_threedee.annotators.surfaces.validation import validate_layer, \
    validate_n3d_zarr


def test_single_surface_instantiation():
    surface_data = [np.random.uniform(0, 10, size=(10, 3))]
    surface = N3dSurface(data=surface_data)
    assert isinstance(surface, N3dSurface)


def test_splines_instantiation():
    surface = N3dSurface(data=[np.random.uniform(0, 10, size=(10, 3))])
    surfaces = N3dSurfaces(data=[surface, surface])
    assert isinstance(surfaces, N3dSurfaces)
    assert surfaces.n_points == 20
    assert surfaces.ndim == 3


def test_splines_to_and_from_layer():
    surface = N3dSurface(data=[np.random.uniform(0, 10, size=(10, 3))])
    surfaces = N3dSurfaces(data=[surface, surface])
    layer = surfaces.as_layer()
    validate_layer(layer)
    splines = N3dSurfaces.from_layer(layer)
    assert isinstance(splines, N3dSurfaces)


def test_splines_to_and_from_n3d_zarr(tmp_path):
    surface = N3dSurface(data=[np.random.uniform(0, 10, size=(10, 3))])
    surfaces = N3dSurfaces(data=[surface, surface])
    surfaces.to_n3d_zarr(tmp_path)
    n3d_zarr = zarr.open(tmp_path)
    validate_n3d_zarr(n3d_zarr)
    splines = N3dSurfaces.from_n3d_zarr(tmp_path)
    assert isinstance(splines, N3dSurfaces)
