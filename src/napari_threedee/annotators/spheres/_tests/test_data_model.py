import numpy as np
import zarr

from napari_threedee.data_models import N3dSpheres
from napari_threedee.annotators.spheres.validation import validate_layer, validate_n3d_zarr


def test_n3d_spheres_instantiation():
    centers = np.random.uniform(low=0, high=10, size=(10, 3))
    radii = np.random.uniform(low=0, high=5, size=(10, ))
    spheres = N3dSpheres(centers=centers, radii=radii)
    assert isinstance(spheres, N3dSpheres)


def test_spheres_to_and_from_layer():
    centers = np.random.uniform(low=0, high=10, size=(10, 3))
    radii = np.random.uniform(low=0, high=5, size=(10, ))
    spheres = N3dSpheres(centers=centers, radii=radii)
    layer = spheres.as_layer()
    validate_layer(layer)
    spheres = N3dSpheres.from_layer(layer)
    assert isinstance(spheres, N3dSpheres)


def test_spheres_to_and_from_n3d_zarr(tmp_path):
    centers = np.random.uniform(low=0, high=10, size=(10, 3))
    radii = np.random.uniform(low=0, high=5, size=(10, ))
    spheres = N3dSpheres(centers=centers, radii=radii)
    spheres.to_n3d_zarr(tmp_path)
    n3d_zarr = zarr.open(tmp_path)
    validate_n3d_zarr(n3d_zarr)
    spheres = N3dSpheres.from_n3d_zarr(tmp_path)
    assert isinstance(spheres, N3dSpheres)