from napari_threedee.annotators import SphereAnnotator
from napari_threedee.annotators.spheres import SphereAnnotatorMode, \
    SPHERE_ID_FEATURES_KEY, SPHERE_RADIUS_FEATURES_KEY


def test_sphere_annotator_instantiation(viewer_with_plane_and_points_3d):
    viewer = viewer_with_plane_and_points_3d
    plane_layer = viewer.layers['blobs_3d']
    annotator = SphereAnnotator(
        viewer=viewer,
        image_layer=plane_layer
    )
    assert isinstance(annotator, SphereAnnotator)


def test_sphere_annotator_layer_creation(sphere_annotator):
    assert sphere_annotator.points_layer is not None
    assert sphere_annotator.surface_layer is not None


def test_sphere_annotator_update_current_properties(sphere_annotator):
    sphere_annotator._update_current_properties(sphere_id=1, radius=None)
    assert sphere_annotator.points_layer.current_properties[SPHERE_ID_FEATURES_KEY] == 1

    sphere_annotator._update_current_properties(sphere_id=0, radius=1)
    assert sphere_annotator.points_layer.current_properties[SPHERE_ID_FEATURES_KEY] == 0
    assert sphere_annotator.points_layer.current_properties[SPHERE_RADIUS_FEATURES_KEY] == 1

    # passing none should have no effect
    sphere_annotator._update_current_properties(sphere_id=None, radius=None)
    assert sphere_annotator.points_layer.current_properties[SPHERE_ID_FEATURES_KEY] == 0
    assert sphere_annotator.points_layer.current_properties[SPHERE_RADIUS_FEATURES_KEY] == 1


def test_sphere_annotator_enable_add_mode_side_effects(sphere_annotator):
    # add some data
    sphere_annotator.points_layer.add([1, 2, 3])
    assert sphere_annotator.points_layer.selected_data == {0}

    # change to add mode
    sphere_annotator.mode = SphereAnnotatorMode.ADD

    # check no point selected
    assert sphere_annotator.points_layer.selected_data == set()

    # check sphere id for next point updated
    assert sphere_annotator.points_layer.current_properties[SPHERE_ID_FEATURES_KEY] == 1
