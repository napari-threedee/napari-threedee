import numpy as np
from napari.layers import Points

import napari_threedee.annotators.paths.constants
from napari_threedee.annotators.paths import PathAnnotator


def test_path_annotator_instantiation(make_napari_viewer, blobs_layer_4d_plane):
    viewer = make_napari_viewer(ndisplay=3)
    plane_layer = viewer.add_layer(blobs_layer_4d_plane)
    annotator = PathAnnotator(
        viewer=viewer,
        image_layer=plane_layer,
    )
    assert isinstance(annotator.points_layer, Points)
    assert napari_threedee.annotators.paths.constants.PATH_ID_FEATURES_KEY in annotator.points_layer.features
    assert len(annotator.points_layer.data) == 0


def test_add_point(path_annotator):
    points_layer = path_annotator.points_layer
    label = napari_threedee.annotators.paths.constants.PATH_ID_FEATURES_KEY

    # start empty
    assert len(points_layer.data) == 0

    # add points, make sure filament id in feature table matches the annotator
    points_layer.add([1, 2, 3, 4])
    assert len(points_layer.data) == 1
    assert points_layer.features[label][0] == 0

    # change filamanet_id in annotator, add point, check matches
    points_layer.add([2, 3, 4, 5])
    assert points_layer.features[label][1] == 0
    assert path_annotator.active_path_id == 0

    # create new path, make sure id advanced
    path_annotator.activate_new_path_mode()
    points_layer.add([2, 3, 4, 5])
    assert points_layer.features[label][2] == 1
    assert path_annotator.active_path_id == 1


def test_get_colors(path_annotator):
    """Test getting spline colors from the annotator."""
    # add a point. the first spline gets id 0
    points_layer = path_annotator.points_layer
    points_layer.add([1, 2, 3, 4])

    # create a new spline, this advances the spline id to 1
    path_annotator.activate_new_path_mode()
    points_layer.add([2, 3, 4, 5])

    spline_colors = path_annotator._get_path_colors()
    for spline_id in (0, 1):
        # check that both spline ids are in the colors dict
        assert spline_id in spline_colors


def test_draw_paths(path_annotator):
    """Test drawing paths adds paths with different colors to the viewer."""
    # add a first path to the empty points layer
    points_layer = path_annotator.points_layer
    points_layer.add([1, 2, 3, 4])
    points_layer.add([2, 3, 4, 5])

    # add a second path
    path_annotator.activate_new_path_mode()
    points_layer.add([3, 4, 5, 6])
    points_layer.add([4, 5, 6, 7])

    # add an invalid path with only one point
    path_annotator.activate_new_path_mode()
    points_layer.add([5, 6, 7, 8])

    # explicitly draw paths and check content of shapes layer
    path_annotator._draw_paths()
    shapes_layer = path_annotator.shapes_layer
    assert len(shapes_layer.data) == 2
    assert len(shapes_layer.edge_color) == 2

    # check two paths have different colors
    assert not np.allclose(shapes_layer.edge_color[0], shapes_layer.edge_color[1])
