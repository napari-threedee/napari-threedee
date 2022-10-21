from napari_threedee._backend.manipulator.rotator import RotatorSet, Rotator


def test_rotator_instantiation():
    rotator = Rotator.from_string('x')
    assert isinstance(rotator, Rotator)


def test_rotatorset_instantiation():
    rotators = RotatorSet.from_string('xyz')
    assert isinstance(rotators, RotatorSet)
    assert len(rotators) == 3
