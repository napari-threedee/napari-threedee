import pytest

from napari_threedee.annotators import PointAnnotator, PathAnnotator, \
    SphereAnnotator


@pytest.fixture
def point_annotator(make_napari_viewer, points_layer_4d, blobs_layer_4d_plane):
    viewer = make_napari_viewer(ndisplay=3)
    points_layer = viewer.add_layer(points_layer_4d)
    plane_layer = viewer.add_layer(blobs_layer_4d_plane)
    annotator = PointAnnotator(
        viewer=viewer, image_layer=plane_layer, points_layer=points_layer,
    )
    return annotator


@pytest.fixture
def path_annotator(make_napari_viewer, blobs_layer_4d_plane):
    viewer = make_napari_viewer(ndisplay=3)
    plane_layer = viewer.add_layer(blobs_layer_4d_plane)
    annotator = PathAnnotator(viewer=viewer, image_layer=plane_layer)
    return annotator


@pytest.fixture
def sphere_annotator(viewer_with_plane_3d):
    viewer = viewer_with_plane_3d
    plane_layer = viewer.layers['blobs_3d']
    annotator = SphereAnnotator(viewer=viewer, image_layer=plane_layer)
    return annotator
