from __future__ import annotations

from typing import List

import numpy as np
from napari.utils.geometry import rotation_matrix_from_vectors_3d
from pydantic import BaseModel

from .central_axis import CentralAxis, CentralAxisSet
from .rotator import Rotator, RotatorSet
from .translator import TranslatorSet, Translator


class ManipulatorVisualData(BaseModel):
    """Data required to render a manipulator"""
    line_data: ManipulatorLineData  # vertices, connections, vertex colors, vertex associated axis_ids
    handle_data: ManipulatorHandleData  # points, colors, point associated axis_ids
    selected_axes: List[int]

    class Config:
        arbitrary_types_allowed = True


class ManipulatorLineData(BaseModel):
    """Data required to construct a VisPy line visual and relate it to axes."""
    vertices: np.ndarray  # (n_vertices, 3)
    connections: np.ndarray  # (n_segments, 2) array of indices for vertices in connected segments
    colors: np.ndarray  # (n_vertices, 4) array of per-vertex RGBA colors
    axis_identifiers: np.ndarray  # (n_vertices, ) array of per-vertex axis identifiers

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

    def __add__(self, other: ManipulatorLineData) -> ManipulatorLineData:
        if other == 0:
            return self
        vertices = np.concatenate([self.vertices, other.vertices], axis=0)
        reindexed_connections = other.connections.copy() + len(self.connections)
        connections = np.concatenate([self.connections, reindexed_connections], axis=0)
        colors = np.concatenate([self.colors, other.colors], axis=0)
        axis_ids = np.concatenate([self.axis_identifiers, other.axis_identifiers], axis=0)
        return ManipulatorLineData(
            vertices=vertices, connections=connections, colors=colors, axis_identifiers=axis_ids
        )

    def __radd__(self, other):
        return self.__add__(other)

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

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_translator(cls, translator: Translator):
        return cls(
            points=translator.start_point.reshape((1, 3)),
            colors=translator.axis.color.reshape((1, 4)),
            axis_identifiers=np.asarray(translator.axis.id)
        )

    @classmethod
    def from_rotator(cls, rotator: Rotator):
        return cls(
            points=rotator.handle_point.reshape((1, 3)),
            colors=rotator.axis.color.reshape((1, 4)),
            axis_identifiers=np.asarray(rotator.axis.id)
        )

    def __add__(self, other: ManipulatorHandleData) -> ManipulatorHandleData:
        points = np.concatenate([self.points, other.points], axis=0)
        colors = np.concatenate([self.colors, other.colors], axis=0)
        axis_ids = np.concatenate([self.axis_identifiers, other.axis_identifiers], axis=0)
        return ManipulatorHandleData(points=points, colors=colors, axis_identifiers=axis_ids)
