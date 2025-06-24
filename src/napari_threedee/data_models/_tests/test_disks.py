import napari
import zarr

from napari_threedee.data_models.disks import N3dDisk, N3dDisks
from napari_threedee.annotators.disks.validation import validate_layer, validate_n3d_zarr


def test_single_disk_instantiation():
    disk = N3dDisk(center=(0, 0, 0), normal=(0, 0, 1), radius=1.0)
    assert isinstance(disk, N3dDisk)


def test_disks_instantiation():
    disk = N3dDisk(center=(0, 0, 0), normal=(0, 0, 1), radius=1.0)
    disks = N3dDisks(data=[disk, disk])
    assert isinstance(disks, N3dDisks)
    assert len(disks) == 2


def test_disks_to_and_from_layer():
    disk = N3dDisk(center=(0, 0, 0), normal=(0, 0, 1), radius=1.0)
    disks = N3dDisks(data=[disk, disk])
    layer = disks.as_layer()
    validate_layer(layer)
    disks = N3dDisks.from_layer(layer)
    assert isinstance(disks, N3dDisks)


def test_disks_to_and_from_n3d_zarr(tmp_path):
    disk = N3dDisk(center=(0, 0, 0), normal=(0, 0, 1), radius=1.0)
    disks = N3dDisks(data=[disk, disk])
    disks.to_n3d_zarr(tmp_path)
    n3d_zarr = zarr.open(tmp_path)
    validate_n3d_zarr(n3d_zarr)
    disks_from_zarr = N3dDisks.from_n3d_zarr(tmp_path)
    assert isinstance(disks_from_zarr, N3dDisks)


def test_empty_disks():
    disks = N3dDisks(data=[])
    assert disks.centers.shape == (0, 3)
    assert disks.normals.shape == (0, 3)
    assert disks.radii.shape == (0,)


def test_empty_disks_as_layer():
    disks = N3dDisks(data=[])
    layer = disks.as_layer()
    assert isinstance(layer, napari.layers.Points)


def test_disk_radius_validation():
    try:
        N3dDisk(center=(0, 0, 0), normal=(0, 0, 1), radius=-1.0)
        assert False, "Should have raised ValueError for negative radius"
    except ValueError:
        pass

    try:
        N3dDisk(center=(0, 0, 0), normal=(0, 0, 1), radius=0.0)
        assert False, "Should have raised ValueError for zero radius"
    except ValueError:
        pass