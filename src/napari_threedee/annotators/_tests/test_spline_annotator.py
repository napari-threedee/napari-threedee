from napari.layers import Points

from napari_threedee.annotators.spline_annotator import SplineAnnotator


def test_spline_annotator_instantiation(make_napari_viewer, blobs_layer_4d_plane):
    viewer = make_napari_viewer(ndisplay=3)
    plane_layer = viewer.add_layer(blobs_layer_4d_plane)
    annotator = SplineAnnotator(
        viewer=viewer,
        image_layer=plane_layer,
    )
    assert isinstance(annotator.points_layer, Points)
    assert annotator.SPLINE_ID_COLUMN in annotator.points_layer.features
    assert len(annotator.points_layer.data) == 0


def test_add_point(filament_annotator):
    points_layer = filament_annotator.points_layer
    label = filament_annotator.SPLINE_ID_COLUMN

    # start empty
    assert len(points_layer.data) == 0

    # add points, make sure filament id in feature table matches the annotator
    points_layer.add([1, 2, 3, 4])
    assert len(points_layer.data) == 1
    assert points_layer.features[label][0] == filament_annotator.current_spline_id

    # change filemanet_id in annotator, add point, check matches
    filament_annotator.current_spline_id = 534
    points_layer.add([2, 3, 4, 5])
    assert points_layer.features[label][1] == filament_annotator.current_spline_id

