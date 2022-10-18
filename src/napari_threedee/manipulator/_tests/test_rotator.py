from napari_threedee.manipulator.rotator import RotatorSet, Rotator
from napari_threedee.manipulator.axis_model import AxisSet


def test_rotator_instantiation():
    rotator = Rotator.from_string('x')
    assert isinstance(rotator, Rotator)


def test_rotatorset_instantiation():
    rotators = RotatorSet.from_string('xyz')
    assert isinstance(rotators, RotatorSet)
    assert len(rotators) == 3
