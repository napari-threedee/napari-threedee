from napari_threedee._backend.manipulator.central_axis import CentralAxis, CentralAxisSet


def test_central_axis_model_instantiation():
    central_axis = CentralAxis.from_string('x')
    assert isinstance(central_axis, CentralAxis)


def test_central_axis_set_instantiation():
    central_axis_set = CentralAxisSet.from_string('xyz')
    assert isinstance(central_axis_set, CentralAxisSet)
