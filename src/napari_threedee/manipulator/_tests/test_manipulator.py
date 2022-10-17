import numpy as np
import pytest
from napari_threedee.manipulator.manipulator import ManipulatorModel


def test_instantiation():
    manipulator = ManipulatorModel(
        central_axes='xyz',
        translators='xy',
        rotators='z',
    )
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