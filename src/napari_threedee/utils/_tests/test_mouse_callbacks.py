from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from napari_threedee.utils.mouse_callbacks import on_mouse_alt_click_add_point_on_plane

@dataclass
class MockMouseEvent:
    modifiers: List[str]

def test_add_point_on_plane_3d(viewer_with_plane_and_points_3d):
    """Test that points added to a plane are in the proper coordinate
    Note: In this test all the layers have the same, default scale [1, 1, 1]
    """

    viewer = viewer_with_plane_and_points_3d
    points_layer = viewer.layers['Points']
    plane_layer = viewer.layers['blobs_3d']
    viewer.dims.ndisplay = 3

    event = MockMouseEvent(
        modifiers = ['Alt']
    )
    viewer.cursor.position = (14, 14, 14)
    viewer.camera.set_view_direction((1, 0, 0))

    on_mouse_alt_click_add_point_on_plane(
        viewer=viewer_with_plane_and_points_3d,
        event=event,
        points_layer=points_layer,
        image_layer=plane_layer,
    )
    assert len(points_layer.data) == 1
    np.testing.assert_array_almost_equal(points_layer.data[0], (14, 14, 14))


def test_add_point_on_plane_4d(viewer_with_plane_and_points_4d):
    """Test that points added to a plane are in the proper coordinate
    In this test all the layers have the same, default scale [1, 1, 1, 1]
    """

    viewer = viewer_with_plane_and_points_4d
    points_layer = viewer.layers['Points']
    plane_layer = viewer.layers['blobs_4d']
    viewer.dims.ndisplay = 3

    # set the dims point
    slice_index = 12
    viewer.dims.set_current_step(0, slice_index)

    event = MockMouseEvent(
        modifiers = ['Alt']
    )
    viewer.cursor.position = (0, 14, 14, 14)
    viewer.camera.set_view_direction((1, 0, 0))

    on_mouse_alt_click_add_point_on_plane(
        viewer=viewer_with_plane_and_points_4d,
        event=event,
        points_layer=points_layer,
        image_layer=plane_layer,
    )
    assert len(points_layer.data) == 1
    
    np.testing.assert_array_almost_equal(points_layer.data[0], (slice_index, 14, 14, 14))


def test_add_point_on_plane_same_scale_3d(viewer_with_plane_and_points_3d):
    """Test adding points on a plane when the layers have same non-[1, 1, 1] scale"""
    viewer = viewer_with_plane_and_points_3d
    scale = (2, .5, .5)
    points_layer = viewer.layers['Points']
    points_layer.scale = scale
    plane_layer = viewer.layers['blobs_3d']
    plane_layer.scale = scale
    viewer.dims.ndisplay = 3

    # the event is in world (scaled) coordinates
    event = MockMouseEvent(
        modifiers = ['Alt']
    )
    viewer.cursor.position = (14, 14, 14)
    viewer.camera.set_view_direction((1, 0, 0))

    # plane position is (14, 14, 14), in data coordinates 
    on_mouse_alt_click_add_point_on_plane(
        viewer=viewer_with_plane_and_points_3d,
        event=event,
        points_layer=points_layer,
        image_layer=plane_layer,
    )
    assert len(points_layer.data) == 1

    # check the point, it should be in Point data coordinates
    # because scales are the same, the Point will be located at
    # plane.position z-slice and de-scaled even.position y, x
    expected_point = np.array([[14, 28, 28]])
    actual_points = points_layer.data
    np.testing.assert_array_equal(actual_points, expected_point)


def test_add_point_on_plane_same_scale_4d(viewer_with_plane_and_points_4d):
    """Test adding points on a plane when the layers have same non-[1, 1, 1] scale"""
    viewer = viewer_with_plane_and_points_4d
    scale = (1, 2, .5, .5)
    points_layer = viewer.layers['Points']
    points_layer.scale = scale
    plane_layer = viewer.layers['blobs_4d']
    plane_layer.scale = scale
    viewer.dims.ndisplay = 3

    # set the dims point
    slice_index = 12
    viewer.dims.set_current_step(0, slice_index)

    # the event is in world (scaled) coordinates
    event = MockMouseEvent(
        modifiers = ['Alt']
    )
    viewer.cursor.position = (0, 14, 14, 14)
    viewer.camera.set_view_direction((1, 0, 0))

    # plane position is (14, 14, 14), in data coordinates 
    on_mouse_alt_click_add_point_on_plane(
        viewer=viewer_with_plane_and_points_4d,
        event=event,
        points_layer=points_layer,
        image_layer=plane_layer,
    )
    assert len(points_layer.data) == 1

    # check the point, it should be in Point data coordinates
    # because scales are the same, the Point will be located at
    # plane.position z-slice and de-scaled even.position y, x
    expected_point = np.array([[12, 14, 28, 28]])
    actual_points = points_layer.data
    np.testing.assert_array_equal(actual_points, expected_point)


def test_add_point_on_plane_different_scale_3d(make_napari_viewer):
    """Test adding points on a plane when the layers don't have the same scale"""
    # set up the viewer
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3

    # set up image layer
    rng = np.random.default_rng(seed=42)
    image = rng.random((10, 40, 40))
    image_layer = viewer.add_image(
        image,
        scale=(2, 0.5, 0.5),
        depiction="plane",
        plane={
            # data (pixel) coordinates
            'position': (5, 10, 10),
            'normal': (1, 0, 0),
            'thickness': 10,
        },
    )

    # set up points layer
    points_layer = viewer.add_points(ndim=3, scale=(1, 1, 1))

    # add the point
    event = MockMouseEvent(
        # world (scaled) coordinates
        modifiers=["Alt"]
    )
    viewer.cursor.position = (12, 5, 5)
    viewer.camera.set_view_direction((-1, 0, 0))

    on_mouse_alt_click_add_point_on_plane(
        viewer=viewer,
        event=event,
        points_layer=points_layer,
        image_layer=image_layer,
        replace_selected=False,
    )

    # check the point, it will be in Point data coordinates
    # The (scaled) world coords will be de-scaled by Points scale [1, 1, 1], so 
    # the world position on the plane should be returned
    expected_point = np.array([[10, 5, 5]])
    actual_points = points_layer.data
    np.testing.assert_array_equal(actual_points, expected_point)
