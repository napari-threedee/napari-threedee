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
