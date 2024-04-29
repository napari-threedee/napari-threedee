import numpy as np
import pytest
from napari_threedee._backend.manipulator.manipulator_model import ManipulatorModel
from napari_threedee.manipulators import RenderPlaneManipulator


def test_instantiation():
    manipulator = ManipulatorModel(
        central_axes='xyz',
        translators='xy',
        rotators='z',
    )
    assert isinstance(manipulator, ManipulatorModel)


def test_instantiation_with_empty_rotators():
    manipulator = ManipulatorModel.from_strings(translators='xyz', rotators=None)
    assert isinstance(manipulator, ManipulatorModel)


def test_instantiation_with_empty_translators():
    manipulator = ManipulatorModel.from_strings(translators=None, rotators='xyz')
    assert isinstance(manipulator, ManipulatorModel)


@pytest.mark.xfail
def test_rotation_matrix_type():
    manipulator = ManipulatorModel(central_axes='xyz', rotators='xyz', translators='xyz')
    assert type(manipulator.rotation_matrix) == np.ndarray


def test_instantiation_from_strings():
    """Central axes are implied by set of translators and rotators."""
    manipulator = ManipulatorModel.from_strings(translators='x', rotators='z')
    central_axes = str(manipulator.central_axes)
    assert len(central_axes) == 2
    assert 'x' in central_axes
    assert 'y' in central_axes
    assert 'z' not in central_axes


def test_selected_object_type_update():
    """Selected object type should be set to None if selected axis id is unset."""
    manipulator = ManipulatorModel.from_strings(translators='x', rotators='z')
    manipulator.selected_axis_id = 1
    manipulator.selected_object_type = 'rotator'
    assert manipulator.selected_object_type == 'rotator'
    manipulator.selected_axis_id = None
    assert manipulator.selected_object_type is None


def test_depiction_change(viewer_with_plane_3d):
    """Test the manipulator is disabled when switching to volume depiction."""
    viewer = viewer_with_plane_3d
    manipulator = RenderPlaneManipulator(viewer=viewer, layer=viewer.layers[0])

    assert viewer.layers[0].depiction == 'plane'
    assert manipulator.enabled

    viewer.layers[0].depiction = 'volume'
    assert manipulator.enabled is False

    
def test_ndisplay_change(viewer_with_plane_3d):
    """Test the manipulator is disabled when switching to 2D display."""
    viewer = viewer_with_plane_3d
    manipulator = RenderPlaneManipulator(viewer=viewer, layer=viewer.layers[0])

    assert viewer.dims.ndisplay == 3
    assert manipulator.enabled

    viewer.dims.ndisplay = 2
    assert manipulator.enabled is False

def test_radius_setter(viewer_with_plane_3d):
    """Test that the radius setter affects the components as expected."""
    viewer = viewer_with_plane_3d
    manipulator = RenderPlaneManipulator(viewer=viewer, layer=viewer.layers[0])

    assert manipulator.radius == 20
    for translator in manipulator._backend.manipulator_model.translators:
        assert translator.distance_from_origin == 20
    for rotator in manipulator._backend.manipulator_model.rotators:
        assert rotator.distance_from_origin == 20
    for axis in manipulator._backend.manipulator_model.central_axes:
        assert axis.length == 20

    manipulator.radius = 50
    for translator in manipulator._backend.manipulator_model.translators:
        assert translator.distance_from_origin == 50
    for rotator in manipulator._backend.manipulator_model.rotators:
        assert rotator.distance_from_origin == 50
    for axis in manipulator._backend.manipulator_model.central_axes:
        assert axis.length == 50

def test_handle_size_setter(viewer_with_plane_3d):
    """Test that the handle_size setter affects the components as expected."""
    viewer = viewer_with_plane_3d
    manipulator = RenderPlaneManipulator(viewer=viewer, layer=viewer.layers[0])

    assert manipulator.handle_size == 10
    for translator in manipulator._backend.manipulator_model.translators:
        assert translator.handle_size == 10
    for rotator in manipulator._backend.manipulator_model.rotators:
        assert rotator.handle_size == 10

    manipulator.handle_size = 20
    for translator in manipulator._backend.manipulator_model.translators:
        assert translator.handle_size == 20
    for rotator in manipulator._backend.manipulator_model.rotators:
        assert rotator.handle_size == 20
