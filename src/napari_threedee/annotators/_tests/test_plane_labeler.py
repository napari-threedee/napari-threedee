from napari.layers.labels._labels_mouse_bindings import draw as napari_draw

from napari_threedee.annotators.label import PlaneLabeler


def test_plane_labeler(viewer_with_plane_and_labels_3d):

    image_layer = viewer_with_plane_and_labels_3d.layers["blobs_3d"]
    labels_layer = viewer_with_plane_and_labels_3d.layers["labels_3d"]
    labeler = PlaneLabeler(viewer=viewer_with_plane_and_labels_3d, image_layer=image_layer, labels_layer=labels_layer)

    labels_layer.mode = "paint"
    assert napari_draw in labels_layer.mouse_drag_callbacks

    labeler.enabled = True
    assert napari_draw not in labels_layer.mouse_drag_callbacks
    assert labeler.draw in labels_layer.mouse_drag_callbacks

    labels_layer.mode = "fill"
    assert labeler.draw not in labels_layer.mouse_drag_callbacks
    assert napari_draw in labels_layer.mouse_drag_callbacks

    labels_layer.mode = "erase"
    assert napari_draw not in labels_layer.mouse_drag_callbacks
    assert labeler.draw in labels_layer.mouse_drag_callbacks

    labeler.enabled = False
    assert napari_draw in labels_layer.mouse_drag_callbacks