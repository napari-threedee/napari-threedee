import pytest

from napari_threedee.annotators import PlanePointAnnotator, SplineAnnotator


@pytest.fixture
def plane_point_annotator(make_napari_viewer, points_layer_4d, blobs_layer_4d_plane):
    viewer = make_napari_viewer(ndisplay=3)
    points_layer = viewer.add_layer(points_layer_4d)
    plane_layer = viewer.add_layer(blobs_layer_4d_plane)
    annotator = PlanePointAnnotator(
        viewer=viewer, image_layer=plane_layer, points_layer=points_layer,
    )
    return annotator


@pytest.fixture
def filament_annotator(make_napari_viewer, blobs_layer_4d_plane):
    viewer = make_napari_viewer(ndisplay=3)
    plane_layer = viewer.add_layer(blobs_layer_4d_plane)
    annotator = SplineAnnotator(viewer=viewer, image_layer=plane_layer)
    return annotator