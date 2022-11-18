from typing import List, Optional

import numpy as np
from napari.utils.geometry import rotation_matrix_from_vectors_3d
from pydantic import BaseModel

from .central_axis import CentralAxis, CentralAxisSet
from .manipulator_model import ManipulatorModel
from .rotator import Rotator, RotatorSet
from .translator import TranslatorSet, Translator
from .utils import ModelWithSettableProperties


class ManipulatorLineData(BaseModel):
    """Data required to construct a VisPy line visual and relate it to axes."""
    vertices: np.ndarray  # (n_vertices, 3)
    connections: np.ndarray  # (n_segments, 2) array of indices for vertices in connected segments
    colors: np.ndarray  # (n_vertices, 4) array of per-vertex RGBA colors
    axis_identifiers: np.ndarray  # (n_vertices, ) array of per-vertex axis identifiers

    line_width: float = 3

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_central_axis(cls, central_axis: CentralAxis):
        return cls(
            vertices=central_axis.points,
            connections=np.array([[0, 1]]),
            colors=np.tile(central_axis.axis.color, (2, 1)),
            axis_identifiers=np.array([central_axis.axis.id for _ in central_axis.points])
        )

    @classmethod
    def from_central_axis_set(cls, axes: CentralAxisSet):
        return sum(cls.from_central_axis(axis) for axis in axes)

    @classmethod
    def from_translator(cls, translator: Translator):
        return cls(
            vertices=translator.points,
            connections=np.array([[0, 1]]),
            colors=np.tile(translator.axis.color, (2, 1)),
            axis_identifiers=np.array([translator.axis.id for _ in translator.points])
        )

    @classmethod
    def from_translator_set(cls, translators: TranslatorSet):
        return sum(cls.from_translator(translator) for translator in translators)

    @classmethod
    def from_rotator(cls, rotator: Rotator, n_segments: int = 64):
        r, n, axes = rotator.distance_from_origin, n_segments, rotator.axis.perpendicular_axes
        t = np.linspace(0, np.pi / 2, num=n + 1)
        vertices = np.stack([0 * t, r * np.sin(t), r * np.cos(t)], axis=1).astype(np.float32)
        rotation_matrix = rotation_matrix_from_vectors_3d(rotator.axis.vector, np.array([1, 0, 0]))
        rotated_vertices = vertices @ rotation_matrix.T
        # connect vertices [[0, 1], [1, 2], [2, 3]...]
        segment_start_indices = np.arange(n)
        connections = np.column_stack([segment_start_indices, segment_start_indices + 1])
        colors = np.row_stack([rotator.axis.color for _ in vertices])
        axis_ids = np.array([rotator.axis.id for _ in rotated_vertices])
        return cls(
            vertices=rotated_vertices,
            connections=connections,
            colors=colors,
            axis_identifiers=axis_ids
        )

    @classmethod
    def from_rotator_set(cls, rotators: RotatorSet, n_segments: int = 64):
        return sum(cls.from_rotator(rotator, n_segments=n_segments) for rotator in rotators)

    def __add__(self, other):
        if other == 0 or other is None:
            return self
        vertices = np.concatenate([self.vertices, other.vertices], axis=0)
        reindexed_connections = other.connections + len(self.vertices)
        connections = np.concatenate([self.connections, reindexed_connections], axis=0)
        colors = np.concatenate([self.colors, other.colors], axis=0)
        axis_ids = np.concatenate([self.axis_identifiers, other.axis_identifiers], axis=0)
        return ManipulatorLineData(
            vertices=vertices, connections=connections, colors=colors, axis_identifiers=axis_ids
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __len__(self):
        return len(self.vertices)

    @property  # move to manipulator
    def rendered_colors(self) -> np.ndarray:
        attenuation_factor = 0.5
        if len(self.highlighted) == 0:
            rendered_axis_colors = self.vertex_colors
        else:
            highlighted_axis_mask = self.axis_vertex_mask(self.highlighted)
            rendered_axis_colors = attenuation_factor * self.vertex_colors
            rendered_axis_colors[highlighted_axis_mask] = self.vertex_colors[highlighted_axis_mask]

        return rendered_axis_colors


class ManipulatorHandleData(BaseModel):
    """Data required to construct a VisPy points visuals and relate points to axes."""
    points: np.ndarray  # (n_handles, 3) array of points
    colors: np.ndarray  # (n_handles, 4) array of RGBA colors for points
    axis_identifiers: np.ndarray  # (n_points, ) array of axis identifiers

    handle_size: float = 10

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_translator(cls, translator: Translator):
        return cls(
            points=translator.start_point.reshape((1, 3)),
            colors=translator.axis.color.reshape((1, 4)),
            axis_identifiers=np.array([translator.axis.id])
        )

    @classmethod
    def from_translator_set(cls, translators: TranslatorSet):
        return sum(cls.from_translator(translator) for translator in translators)

    @classmethod
    def from_rotator(cls, rotator: Rotator):
        return cls(
            points=rotator.handle_point.reshape((1, 3)),
            colors=rotator.axis.color.reshape((1, 4)),
            axis_identifiers=np.array([rotator.axis.id])
        )

    @classmethod
    def from_rotator_set(cls, rotators: RotatorSet):
        return sum(cls.from_rotator(rotator) for rotator in rotators)

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if other == 0 or other is None:
            return self
        points = np.concatenate([self.points, other.points], axis=0)
        colors = np.concatenate([self.colors, other.colors], axis=0)
        axis_ids = np.concatenate([self.axis_identifiers, other.axis_identifiers], axis=0)
        return ManipulatorHandleData(points=points, colors=colors, axis_identifiers=axis_ids)

    def __len__(self):
        return len(self.points)


class ManipulatorVisualData(ModelWithSettableProperties):
    """Data required to render a manipulator"""
    central_axis_line_data: Optional[ManipulatorLineData]
    translator_line_data: Optional[ManipulatorLineData]
    translator_handle_data: Optional[ManipulatorHandleData]
    rotator_line_data: Optional[ManipulatorLineData]
    rotator_handle_data: Optional[ManipulatorHandleData]

    selected_axes: List[int] = []
    translator_is_selected: bool = False

    _attenuation_factor: float = 0.5

    class Config:
        arbitrary_types_allowed = True

    @property
    def rotator_is_selected(self) -> bool:
        return not self.translator_is_selected

    @rotator_is_selected.setter
    def rotator_is_selected(self, value: bool):
        self.translator_is_selected = not value

    @classmethod
    def from_manipulator(cls, manipulator: ManipulatorModel):
        central_axis_line_data = ManipulatorLineData.from_central_axis_set(manipulator.central_axes)
        # translators
        if manipulator.translators is None:
            translator_line_data = None
            translator_handle_data = None
        else:
            translator_line_data = ManipulatorLineData.from_translator_set(manipulator.translators)
            translator_handle_data = ManipulatorHandleData.from_translator_set(manipulator.translators)
        # rotators
        if manipulator.rotators is None:
            rotator_line_data = None
            rotator_handle_data = None
        else:
            rotator_line_data = ManipulatorLineData.from_rotator_set(manipulator.rotators)
            rotator_handle_data = ManipulatorHandleData.from_rotator_set(manipulator.rotators)
        return cls(
            central_axis_line_data=central_axis_line_data,
            translator_line_data=translator_line_data,
            translator_handle_data=translator_handle_data,
            rotator_line_data=rotator_line_data,
            rotator_handle_data=rotator_handle_data
        )

    @property
    def central_axis_line_colors(self) -> np.ndarray:
        """Central axis line vertex colors, non-selected axes are attenuated."""
        if self.selected_axes == []:
            central_axis_colors = self.central_axis_line_data.colors
        else:
            central_axis_colors = self.central_axis_line_data.colors.copy()
            central_axis_colors[~self._selected_central_axis_vertices] *= self._attenuation_factor
        return central_axis_colors

    @property
    def translator_line_colors(self) -> np.ndarray:
        """Translator line vertex colors, non-selected axes are attenuated."""
        if self.selected_axes == []:
            translator_line_colors = self.translator_line_data.colors
        else:
            translator_line_colors = self.translator_line_data.colors.copy()
            translator_line_colors[
                ~self._selected_translator_line_vertices] *= self._attenuation_factor
        return translator_line_colors

    @property
    def translator_handle_colors(self) -> np.ndarray:
        """Translator handle colors, non-selected axes are attenuated."""
        if self.selected_axes == []:
            translator_handle_colors = self.translator_handle_data.colors
        else:
            translator_handle_colors = self.translator_handle_data.colors.copy()
            translator_handle_colors[
                ~self._selected_translator_handle_points] *= self._attenuation_factor
        return translator_handle_colors

    @property
    def rotator_line_colors(self) -> np.ndarray:
        """Rotator line vertex colors, non-selected axes are attenuated."""
        if self.selected_axes == []:
            rotator_line_colors = self.rotator_line_data.colors
        else:
            rotator_line_colors = self.rotator_line_data.colors.copy()
            rotator_line_colors[~self._selected_rotator_line_vertices] *= self._attenuation_factor
        return rotator_line_colors

    @property
    def rotator_handle_colors(self) -> np.ndarray:
        """Rotator handle colors, non-selected axes are attenuated."""
        if self.selected_axes == []:
            rotator_handle_colors = self.rotator_handle_data.colors
        else:
            rotator_handle_colors = self.rotator_handle_data.colors.copy()
            rotator_handle_colors[~self._selected_rotator_handle_points] *= self._attenuation_factor
        return rotator_handle_colors

    @property
    def _selected_central_axis_vertices(self) -> np.ndarray:
        if not self.translator_is_selected:
            return np.zeros(len(self.central_axis_line_data)).astype(bool)
        return np.isin(self.central_axis_line_data.axis_identifiers,
                       test_elements=self.selected_axes)

    @property
    def _selected_translator_line_vertices(self) -> np.ndarray:
        if not self.translator_is_selected:
            return np.zeros(len(self.translator_line_data)).astype(bool)
        return np.isin(self.translator_line_data.axis_identifiers, test_elements=self.selected_axes)

    @property
    def _selected_translator_handle_points(self) -> np.ndarray:
        if not self.translator_is_selected:
            return np.zeros(len(self.translator_handle_data)).astype(bool)
        return np.isin(self.translator_handle_data.axis_identifiers,
                       test_elements=self.selected_axes)

    @property
    def _selected_rotator_line_vertices(self) -> np.ndarray:
        if not self.rotator_is_selected:
            return np.zeros(len(self.rotator_line_data)).astype(bool)
        return np.isin(self.rotator_line_data.axis_identifiers, test_elements=self.selected_axes)

    @property
    def _selected_rotator_handle_points(self) -> np.ndarray:
        if not self.rotator_is_selected:
            return np.zeros(len(self.rotator_handle_data)).astype(bool)
        return np.isin(self.rotator_handle_data.axis_identifiers, test_elements=self.selected_axes)
