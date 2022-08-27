"""pytest fixtures which should be available throughout the package."""
import numpy as np
import pytest
import skimage

from napari.layers import Image, Points


@pytest.fixture
def blobs_3d() -> np.ndarray:
    blobs = skimage.data.binary_blobs(
        length=28,
        volume_fraction=0.1,
        n_dim=3,
    ).astype(float)
    return blobs


@pytest.fixture
def blobs_4d() -> np.ndarray:
    blobs = skimage.data.binary_blobs(
        length=28,
        volume_fraction=0.1,
        n_dim=4,
    ).astype(float)
    return blobs


@pytest.fixture
def blobs_layer_3d_volume(blobs_3d) -> Image:
    return Image(blobs_3d, depiction='volume')


@pytest.fixture
def blobs_layer_3d_plane(blobs_3d) -> Image:
    plane_parameters = {'position': (14, 14, 14), 'normal': (1, 0, 0)}
    return Image(blobs_3d, depiction='plane', plane=plane_parameters)


@pytest.fixture
def blobs_layer_4d_volume(blobs_4d) -> Image:
    return Image(blobs_4d, depiction='volume')


@pytest.fixture
def blobs_layer_4d_plane(blobs_4d) -> Image:
    plane_parameters = {'position': (14, 14, 14), 'normal': (1, 0, 0)}
    return Image(blobs_4d, depiction='plane', plane=plane_parameters)


@pytest.fixture
def points_layer_3d() -> Points:
    return Points(data=[], ndim=3)


@pytest.fixture
def points_layer_4d() -> Points:
    return Points(data=[], ndim=4)


@pytest.fixture
def viewer_with_plane_and_points_3d(make_napari_viewer, blobs_layer_3d_plane, points_layer_3d):
    viewer = make_napari_viewer(ndisplay=3)
    viewer.add_layer(blobs_layer_3d_plane)
    viewer.add_layer(points_layer_3d)
    return viewer


@pytest.fixture
def viewer_with_plane_and_points_4d(make_napari_viewer, blobs_layer_4d_plane, points_layer_4d):
    viewer = make_napari_viewer(ndisplay=3)
    viewer.add_layer(blobs_layer_4d_plane)
    viewer.add_layer(points_layer_4d)
    return viewer
