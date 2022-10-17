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

