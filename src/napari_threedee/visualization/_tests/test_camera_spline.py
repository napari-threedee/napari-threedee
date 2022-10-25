import napari

from napari_threedee.visualization.camera_spline import CameraSpline, CameraSplineMode


def test_initialize_camera_spline():
    viewer = napari.Viewer()
    camera_spline = CameraSpline(viewer=viewer)

    assert camera_spline.image_layer is None
    assert camera_spline.enabled is False


def test_setting_camera_spline_mode():
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