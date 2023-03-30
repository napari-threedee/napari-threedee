import numpy as np
import pytest

from napari_threedee.visualization.lighting_control import LightingControl
from napari_threedee.utils.napari_utils import get_napari_visual


@pytest.fixture
def lone_triangle():
    vertices = np.array([
        [-1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ])
    faces = np.array([[0, 1, 2]], dtype=np.int16)
    return (vertices, faces)


def test_lighting_control_enable(make_napari_viewer, lone_triangle):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = viewer.add_surface(lone_triangle)

    lightting_control = LightingControl(viewer)

    lightting_control.set_layers([layer])
    assert lightting_control.selected_layers == [layer]

    lightting_control.enabled = True
    assert lightting_control.enabled


def test_light_dir_unchanged_when_disabled(make_napari_viewer, lone_triangle):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = viewer.add_surface(lone_triangle)
    visual = get_napari_visual(viewer=viewer, layer=layer)

    lightting_control = LightingControl(viewer)
    lightting_control.set_layers([layer])

    inital_camera_angles = (0, 0, 90)
    viewer.camera.angles = inital_camera_angles
    inital_light_dir = visual.node.shading_filter.light_dir
    viewer.camera.angles = (0, 0, -90)

    assert visual.node.shading_filter.light_dir == inital_light_dir


def test_light_dir_changed_when_enabled(make_napari_viewer, lone_triangle):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = viewer.add_surface(lone_triangle)
    visual = get_napari_visual(viewer=viewer, layer=layer)

    lightting_control = LightingControl(viewer)
    lightting_control.set_layers([layer])
    lightting_control.enabled = True

    inital_camera_angles = (0, 0, 90)
    viewer.camera.angles = inital_camera_angles
    inital_light_dir = visual.node.shading_filter.light_dir
    viewer.camera.angles = (0, 0, -90)

    assert visual.node.shading_filter.light_dir != inital_light_dir
