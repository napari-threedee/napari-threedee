from napari.layers import Points

from napari_threedee.annotators.filament_annotator import FilamentAnnotator


def test_filament_annotator_instantiation(make_napari_viewer, blobs_layer_4d_plane):
    viewer = make_napari_viewer(ndisplay=3)
    plane_layer = viewer.add_layer(blobs_layer_4d_plane)
    annotator = FilamentAnnotator(
        viewer=viewer,
        image_layer=plane_layer,
    )
    assert isinstance(annotator.points_layer, Points)
    assert annotator.FILAMENT_ID_LABEL in annotator.points_layer.features
    assert len(annotator.points_layer.data) == 0


def test_add_point(filament_annotator):
    points_layer = filament_annotator.points_layer
    label = filament_annotator.FILAMENT_ID_LABEL

    # start empty
    assert len(points_layer.data) == 0

    # add points, make sure filament id in feature table matches the annotator
    points_layer.add([1, 2, 3, 4])
    assert len(points_layer.data) == 1
    assert points_layer.features[label][0] == filament_annotator.current_filament_id

    # change filemanet_id in annotator, add point, check matches
    filament_annotator.current_filament_id = 534
    points_layer.add([2, 3, 4, 5])
    assert points_layer.features[label][1] == filament_annotator.current_filament_id

