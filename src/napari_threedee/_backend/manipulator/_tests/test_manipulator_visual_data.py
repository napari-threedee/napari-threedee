import numpy as np

from napari_threedee._backend.manipulator.axis_model import AxisModel
from napari_threedee._backend.manipulator.central_axis import CentralAxis, CentralAxisSet
from napari_threedee._backend.manipulator.manipulator_model import ManipulatorModel
from napari_threedee._backend.manipulator.translator import Translator, TranslatorSet
from napari_threedee._backend.manipulator.rotator import Rotator, RotatorSet
from napari_threedee._backend.manipulator.vispy_visual_data import ManipulatorLineData, \
    ManipulatorHandleData, ManipulatorVisualData


def test_linedata_from_central_axis():
    ld = ManipulatorLineData.from_central_axis(CentralAxis.from_string('x'))
    assert isinstance(ld, ManipulatorLineData)
    assert ld.vertices.shape == (2, 3)  # 2 vertices per axis
    assert ld.connections.shape == (1, 2)  # 1 segment per axis
    assert len(ld.colors) == len(ld.vertices)  # 1 color per vertex


def test_linedata_from_central_axis_set():
    ld = ManipulatorLineData.from_central_axis_set(CentralAxisSet.from_string('xyz'))
    assert isinstance(ld, ManipulatorLineData)
    assert ld.vertices.shape == (6, 3)  # 2 vertices per axis
    assert ld.connections.shape == (3, 2)  # 1 segment per axis
    assert ld.colors.shape == (6, 4)  # 1 color per vertex


def test_linedata_from_rotator():
    n_segments = 64
    ld = ManipulatorLineData.from_rotator(Rotator.from_string('x'), n_segments=64)
    assert isinstance(ld, ManipulatorLineData)
    n_expected_vertices = (n_segments + 1)
    assert ld.vertices.shape == (n_expected_vertices, 3)
    assert ld.connections.shape == (n_segments, 2)
    assert ld.colors.shape == (n_expected_vertices, 4)
    assert ld.axis_identifiers.shape == (n_expected_vertices,)


def test_linedata_from_rotator_set():
    n_segments, n_rotators = 64, 3
    ld = ManipulatorLineData.from_rotator_set(RotatorSet.from_string('xyz'))
    assert isinstance(ld, ManipulatorLineData)
    n_expected_vertices = (n_segments + 1) * n_rotators
    assert ld.vertices.shape == (n_expected_vertices, 3)
    assert ld.connections.shape == (n_segments * n_rotators, 2)
    assert ld.colors.shape == (n_expected_vertices, 4)
    assert ld.axis_identifiers.shape == (n_expected_vertices,)


def test_linedata_from_translator():
    ld = ManipulatorLineData.from_translator(Translator.from_string('x'))
    assert isinstance(ld, ManipulatorLineData)
    n_expected_vertices = 2
    n_expected_connections = 1
    assert ld.vertices.shape == (n_expected_vertices, 3)
    assert ld.connections.shape == (n_expected_connections, 2)
    assert ld.colors.shape == (n_expected_vertices, 4)
    assert ld.axis_identifiers.shape == (n_expected_vertices,)


def test_linedata_from_translator_set():
    n_translators = 3
    ld = ManipulatorLineData.from_translator_set(TranslatorSet.from_string('xyz'))
    assert isinstance(ld, ManipulatorLineData)
    n_expected_vertices = 2 * n_translators
    n_expected_connections = n_translators
    assert ld.vertices.shape == (n_expected_vertices, 3)
    assert ld.connections.shape == (n_expected_connections, 2)
    assert ld.colors.shape == (n_expected_vertices, 4)
    assert ld.axis_identifiers.shape == (n_expected_vertices,)


def test_reindex_on_add_linedata():
    """connections must be reindexed after adding pieces of linedata together"""
    ld_x = ManipulatorLineData.from_central_axis(CentralAxis.from_string('x'))
    ld_y = ManipulatorLineData.from_central_axis(CentralAxis.from_string('y'))
    ld_xy = ld_x + ld_y
    assert len(ld_xy.vertices) == len(ld_x.vertices) + len(ld_y.vertices)
    assert not np.allclose(ld_xy.connections[1], ld_y.connections[0])
    assert np.allclose(ld_xy.connections[1], ld_y.connections[0] + len(ld_x.vertices))


def test_handledata_from_translator():
    hd = ManipulatorHandleData.from_translator(Translator.from_string('x'))
    assert len(hd.points) == 1
    assert np.allclose(hd.axis_identifiers, AxisModel.from_string('x').id)


def test_handledata_from_rotator():
    hd = ManipulatorHandleData.from_rotator(Rotator.from_string('x'))
    assert len(hd.points) == 1
    assert np.allclose(hd.axis_identifiers, AxisModel.from_string('x').id)


def test_handledata_from_translator_set():
    hd = ManipulatorHandleData.from_translator_set(TranslatorSet.from_string('xyz'))
    assert len(hd.points) == 3
    assert len(hd.axis_identifiers) == 3


def test_handledata_from_rotator_set():
    hd = ManipulatorHandleData.from_rotator_set(RotatorSet.from_string('xyz'))
    assert len(hd.points) == 3
    assert len(hd.axis_identifiers) == 3


def test_manipulator_visual_data_from_manipulator():
    manipulator = ManipulatorModel(central_axes='xyz', translators='xyz', rotators='xyz')
    mvd = ManipulatorVisualData.from_manipulator(manipulator=manipulator)
    assert isinstance(mvd, ManipulatorVisualData)
    assert isinstance(mvd.translator_line_data, ManipulatorLineData)
    assert isinstance(mvd.translator_handle_data, ManipulatorHandleData)
