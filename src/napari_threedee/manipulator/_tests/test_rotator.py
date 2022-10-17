from napari_threedee.manipulator.rotator import RotatorSet, Rotator
from napari_threedee.manipulator.axis_model import AxisSet


def test_rotator_instantiation():
    rotator = Rotator.from_string('x')
    assert isinstance(rotator, Rotator)


def test_rotatorset_instantiation():
    aset = AxisSet.from_string('xyz')
    rotators = RotatorSet(axes=aset)
    assert isinstance(rotators, RotatorSet)
    assert len(rotators.axes) == 3
