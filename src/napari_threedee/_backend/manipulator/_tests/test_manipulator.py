import numpy as np
import pytest
from napari_threedee._backend.manipulator.manipulator_model import ManipulatorModel


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
