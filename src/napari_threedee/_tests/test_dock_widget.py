import napari_threedee
import pytest

MY_PLUGIN_NAME = "napari-threedee"
MY_WIDGET_NAMES = [
    "render plane manipulator",
    "point manipulator",
    "layer manipulator",
    "plane point annotator",
    "spline annotator",
    "mesh lighting controls",
    "camera spline control"
]


@pytest.mark.parametrize("widget_name", MY_WIDGET_NAMES)
def test_something_with_viewer(
    widget_name, make_napari_viewer, napari_plugin_manager
):
    napari_plugin_manager.register(napari_threedee, name=MY_PLUGIN_NAME)
    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name=MY_PLUGIN_NAME, widget_name=widget_name
    )
    assert len(viewer.window._dock_widgets) == num_dw + 1
