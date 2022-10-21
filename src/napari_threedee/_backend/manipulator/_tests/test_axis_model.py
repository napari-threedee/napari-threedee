import numpy as np

from napari_threedee._backend.manipulator.axis_model import AxisModel, AxisSet


def test_axis_model_instantiation_from_string():
    a = AxisModel.from_string('x')
    assert a.name == 'x'
    assert np.allclose(a.vector, [0, 0, 1])
    assert np.allclose(a.points, [[0, 0, 0], [0, 0, 1]])
    assert all(ax in a.perpendicular_axes for ax in ('y', 'z'))


def test_axis_set_instantiation_from_string():
    aset = AxisSet.from_string('xyz')
    assert len(aset) == 3


def test_perpendicular_axes():
    axes = AxisModel.from_string('x').perpendicular_axes
    assert len(axes) == 2
    assert 'y' in axes
    assert 'z' in axes



