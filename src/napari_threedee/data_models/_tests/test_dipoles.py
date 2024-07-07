import napari
import zarr

from napari_threedee.data_models.dipoles import N3dDipole, N3dDipoles
from napari_threedee.annotators.dipoles.validation import validate_layer, validate_n3d_zarr


def test_single_dipole_instantiation():
    dipole = N3dDipole(center=(0, 0, 0), direction=(1, 1, 1))
    assert isinstance(dipole, N3dDipole)


def test_dipoles_instantiation():
    dipole = N3dDipole(center=(0, 0, 0), direction=(1, 1, 1))
    dipoles = N3dDipoles(data=[dipole, dipole])
    assert isinstance(dipoles, N3dDipoles)
    assert len(dipoles) == 2


def test_dipoles_to_and_from_layer():
    dipole = N3dDipole(center=(0, 0, 0), direction=(1, 1, 1))
    dipoles = N3dDipoles(data=[dipole, dipole])
    layer = dipoles.as_layer()
    validate_layer(layer)
    dipoles = N3dDipoles.from_layer(layer)
    assert isinstance(dipoles, N3dDipoles)


def test_paths_to_and_from_n3d_zarr(tmp_path):
    dipole = N3dDipole(center=(0, 0, 0), direction=(1, 1, 1))
    dipoles = N3dDipoles(data=[dipole, dipole])
    dipoles.to_n3d_zarr(tmp_path)
    n3d_zarr = zarr.open(tmp_path)
    validate_n3d_zarr(n3d_zarr)
    paths = N3dDipoles.from_n3d_zarr(tmp_path)
    assert isinstance(paths, N3dDipoles)


def test_empty_dipoles():
    dipoles = N3dDipoles(data=[])
    assert dipoles.centers.shape == (0, 3)
    assert dipoles.directions.shape == (0, 3)


def test_empty_dipoles_as_layer():
    dipoles = N3dDipoles(data=[])
    layer = dipoles.as_layer()
    assert isinstance(layer, napari.layers.Points)
