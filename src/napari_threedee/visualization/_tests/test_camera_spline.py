import platform
from typing import Tuple

import napari
import numpy as np
import pytest

from napari_threedee.visualization.camera_spline import CameraSpline, CameraSplineMode


def test_initialize_camera_spline():
    """Test creation of an empty CameraSpline model."""
    viewer = napari.Viewer()
    camera_spline = CameraSpline(viewer=viewer)

    assert camera_spline.image_layer is None
    assert camera_spline.enabled is False


def test_setting_camera_spline_mode():
    """Test setting camera spline mode with strings and Enum"""
    viewer = napari.Viewer()
    camera_spline = CameraSpline(viewer=viewer)

    # default mode
    assert camera_spline.mode == CameraSplineMode.PAN_ZOOM

    # with strings
    camera_spline.mode = "explore"
    assert camera_spline.mode == CameraSplineMode.EXPLORE

    # strings are case-insensitve
    camera_spline.mode = "PaN_ZooM"
    assert camera_spline.mode == CameraSplineMode.PAN_ZOOM

    # with Enum
    camera_spline.mode = CameraSplineMode.ANNOTATE
    assert camera_spline.mode == CameraSplineMode.ANNOTATE


class FakeMouseEvent:
    position: Tuple[float, float, float]
    view_direction: Tuple[float, float, float] = (1, 0, 0)
    dims_displayed: Tuple[int, int, int] = (0, 1, 2)
    modifiers: Tuple[str] = ("Alt",)


@pytest.mark.skipif(platform.system() == 'Windows', reason="fails on CI")
def test_annotate_spline():
    """Test annotating a spline in the CameraSpline model."""
    """Test setting camera spline mode with strings and Enum"""
    viewer = napari.Viewer()
    camera_spline = CameraSpline(viewer=viewer)

    # add an image layer to the viewer
    plane_parameters_z = {
        'position': (15, 15, 15),
        'normal': (1, 0, 0),
        'thickness': 10,
    }
    image_layer = viewer.add_image(np.random.random((30, 30, 30)), plane=plane_parameters_z)

    camera_spline.set_layers(image_layer)
    assert camera_spline.spline_valid is False

    # enter annotation mode
    camera_spline.enabled = True
    camera_spline.mode = "annotate"

    # get the points layer
    points_layer = camera_spline.spline_annotator_model.points_layer
    assert isinstance(points_layer, napari.layers.Points)

    # add 4 points and check that the spline was created
    points_layer.add(np.random.random((4, 3)))
    assert camera_spline.spline_valid is True
    assert len(points_layer.metadata["splines"]) == 1


@pytest.mark.skipif(platform.system() == 'Windows', reason="fails on CI")
def test_spline_explore():
    """Test moving the camera along the spline"""
    viewer = napari.Viewer()
    viewer.dims.ndisplay = 3
    camera_spline = CameraSpline(viewer=viewer)

    # add an image layer to the viewer
    plane_parameters_z = {
        'position': (15, 15, 15),
        'normal': (1, 0, 0),
        'thickness': 10,
    }
    image_layer = viewer.add_image(np.random.random((30, 30, 30)), plane=plane_parameters_z)

    camera_spline.set_layers(image_layer)
    assert camera_spline.spline_valid is False

    # verify that spline position can't be set without valid spline
    initial_spline_coordinate = camera_spline.current_spline_coordinate
    assert initial_spline_coordinate == 0
    camera_spline.set_camera_position(0.5)
    assert initial_spline_coordinate == 0

    # enter annotation mode
    camera_spline.enabled = True
    camera_spline.mode = "annotate"

    # add 4 points and check that the spline was created
    points_layer = camera_spline.spline_annotator_model.points_layer
    points_layer.add(10 * np.random.random((4, 3)))
    assert camera_spline.spline_valid is True

    # enter explore mode and set the camera position
    camera_spline.mode = "explore"
    camera_spline.current_spline_coordinate = 0.5
