from typing import Union, List, Tuple, Optional, Dict

import numpy as np

from .manipulator_utils import make_rotator_data


MANIPULATOR_BASIS = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
)
CENTRAL_AXIS_VERTICES: Dict[int, np.ndarray] = {
    0: np.array([[0, 0, 0], [1, 0, 0]]),
    1: np.array([[0, 0, 0], [0, 1, 0]]),
    2: np.array([[0, 0, 0], [0, 0, 1]]),
}


class CentralAxesModel:
    def __init__(self, normal_vectors: Union[np.ndarray, List[int]], colors: np.ndarray, radius: float):
        self._normal_vector_indices: List[int] = normal_vectors
        self.radius = radius
        self.colors = np.atleast_2d(colors)

        vertices, connections, colors, axis_indices = self._make_axis_data()
        self._vertices = vertices
        self._connections = connections
        self._vertex_colors = colors
        self._axis_indices = axis_indices

        self._highlighted = []

    def _make_axis_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vertices = []
        connections = []
        colors = []
        axis_indices = []
        vertex_offset = 0
        for index, axis_color in zip(self._normal_vector_indices, self.colors):
            vertices.append(self.radius * CENTRAL_AXIS_VERTICES[index])

            axis_connections = vertex_offset + np.array([0, 1])
            connections.append(axis_connections)

            colors.append(np.tile(axis_color, (2, 1)))

            # add the rotator indices
            axis_indices.append([index] * 2)

            vertex_offset += 2
        return np.concatenate(vertices, axis=0), np.stack(connections), np.concatenate(colors, axis=0), np.concatenate(axis_indices)


    @property
    def normal_vectors(self) -> np.ndarray:
        """Normal vectors for the central axis .

        Returns
        -------
        normal_vectors : np.ndarray
            The normal vectors in data coordinates for the untransformed manipulator
        """
        return MANIPULATOR_BASIS[self._normal_vector_indices]

    @property
    def vertices(self) -> np.ndarray:
        """Coordinates for the rotator arc vertices.

        Returns
        -------
        vertices : np.ndarray
            (n_rotators * n_segments, 3) array containing the coordinates
            of all vertices.
        """
        return self._vertices

    @property
    def connections(self) -> np.ndarray:
        """Connections between vertices in the rotator arc.

        Returns
        -------
        connections : np.ndarray
            (n_rotators * [n_segments - 1], 2) array containing the
            connections between arc vertices.
        """
        return self._connections

    @property
    def vertex_colors(self) -> np.ndarray:
        """The color for each vertex.

        Returns
        -------
        vertex_colors : np.ndarray
            (n_rotators * n_segments, 4) array containing RGBA colors
            for all vertices.
        """
        return self._vertex_colors

    @property
    def axis_indices(self) -> np.ndarray:
        """The axis index for each vertex.

        Returns
        -------
        axis_indices : np.ndarray
            (n_axes * 2,) array containing the axis index for each vertex.
        """
        return self._axis_indices

    def axis_vertex_mask(self, axis_indices: Union[List[int], int]) -> np.ndarray:
        """Create a boolean mask to select the vertices from the specified rotators.

        Parameters
        ----------
        axis_indices : Union[List[int], int]
            The indices of the rotators to set to True in the resulting mask.

        Returns
        -------
        vertex_mask : np.ndarray
            The (n_vertices,) array containing True values where the vertices are
            from the selected rotator(s).
        """
        if isinstance(axis_indices, int):
            rotator_indices = [axis_indices]

        return np.isin(self.axis_indices, axis_indices)

    @property
    def highlighted(self) -> List[int]:
        return self._highlighted

    @highlighted.setter
    def highlighted(self, highlighted: Optional[List[int]]) -> None:
        if highlighted is None:
            self._highlighted = []
        else:
            self._highlighted = list(highlighted)

    @property
    def rendered_colors(self) -> np.ndarray:
        attenuation_factor = 0.5
        if len(self.highlighted) == 0:
            rendered_axis_colors = self.vertex_colors
        else:
            highlighted_axis_mask = self.axis_vertex_mask(self.highlighted)
            rendered_axis_colors = attenuation_factor * self.vertex_colors
            rendered_axis_colors[highlighted_axis_mask] = self.vertex_colors[highlighted_axis_mask]

        return rendered_axis_colors


class TranslatorModel:
    def __init__(
            self,
            normal_vectors: Union[np.ndarray, List[int]],
            colors: np.ndarray,
            radius: float = 20,
            width: float = 5,
            handle_size: float = 3,
            axis_offset: float = 1,
            axis_length: float = 3
    ):
        self._normal_vector_indices = normal_vectors
        self.colors = np.atleast_2d(colors)
        self.radius = radius
        self.width = width
        self.handle_size = handle_size
        self._axis_offset = axis_offset
        self._axis_length = axis_length

        vertices, colors, handles_points, handle_colors, translator_indices = self._make_axis_data()

        self._vertices = vertices
        self._vertex_colors = colors
        self._handle_points = handles_points
        self._handle_colors = handle_colors
        self._translator_indices = translator_indices

        # array
        self._highlighted_translators: List[int] = None

    def _make_axis_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        axis_start_coordinate = self.radius + self._axis_offset
        axis_end_coordinate = axis_start_coordinate + self._axis_length

        vertices = []
        handle_points = []
        handle_colors = []
        colors = []
        axis_indices = []
        for index, axis_color in zip(self._normal_vector_indices, self.colors):
            axis_vector = MANIPULATOR_BASIS[index]

            # add the start point for the translator axis
            axis_start_vertex = axis_start_coordinate * axis_vector
            vertices.append(axis_start_vertex)

            # add the end point for the translator axis
            axis_end_vertex = axis_end_coordinate * axis_vector
            vertices.append(axis_end_vertex)

            # add the vertex colors
            colors.append(np.tile(axis_color, (2, 1)))

            # add the coordinate and colors for the translator handle
            handle_points.append(axis_end_vertex)
            handle_colors.append(axis_color)

            # add the axis indices
            axis_indices.append([index] * 2)

        return np.stack(vertices), np.concatenate(colors, axis=0), np.stack(handle_points), np.stack(handle_colors), np.concatenate(axis_indices)

    @property
    def normal_vectors(self) -> np.ndarray:
        """Normal vectors for the translation axes.

        Returns
        -------
        normal_vectors : np.ndarray
            The normal vectors in data coordinates for the untransformed manipulator
        """
        return MANIPULATOR_BASIS[self._normal_vector_indices]

    @property
    def vertices(self) -> np.ndarray:
        """Coordinates for the translator vertices.

        Returns
        -------
        vertices : np.ndarray
            (n_translators * 2, 3) array containing the coordinates
            of all vertices.
        """
        return self._vertices

    @property
    def vertex_colors(self) -> np.ndarray:
        """The color for each vertex.

        Returns
        -------
        vertex_colors : np.ndarray
            (n_translators * 2, 4) array containing RGBA colors
            for all vertices.
        """
        return self._vertex_colors

    @property
    def handle_points(self) -> np.ndarray:
        """The coordinates for each handle point.

        Returns
        -------
        handle_points : np.ndarray
            (n_translators, 3) array containing the coordinates of the handle for
            each rotator.
        """
        return self._handle_points

    @property
    def handle_colors(self) -> np.ndarray:
        """The color for each translator handle.

        Returns
        handle_colors : np.ndarray
            (n_handles, 4) RGBA array containing the color for each translator handle.
        """
        return self._handle_colors

    @property
    def highlighted_translators(self) -> Optional[List[int]]:
        """Which, if any, translators are highlighted.

        Returns
        -------
        highlighted_translators
            The translators that are highlighted.
            Returns None if no translators are highlighted.
            Returns a list with length == 0 if all translators are attenuated.
            Returns a list with length > 0 if some translators are highlighted.
            The list contains the indices for each highlighted translator.
        """
        return self._highlighted_translators

    @highlighted_translators.setter
    def highlighted_translators(self, highlighted_translators: Optional[List[int]]) -> None:
        if highlighted_translators is None:
            self._highlighted_translators = highlighted_translators
        else:
            self._highlighted_translators = list(highlighted_translators)

    @property
    def rendered_translator_colors(self) -> Tuple[np.ndarray, np.ndarray]:
        """"The colors for the translators as they should be rendered taking highlighting into account.

        Returns
        -------
        new_rotator_arc_colors : np.ndarray
            (n_vertices, 4) RGBA array containing the color for each vertex in the translator axes.
        new_rotator_handle_colors : np.ndarray
            (n_handles, 4) RGBA array containing the color for each translator handle.
        """
        attenuation_factor = 0.5
        if self.highlighted_translators is None:
            new_translator_axis_colors = self.vertex_colors
            new_translator_handle_colors = self.handle_colors
        else:
            highlight_mask = self.translator_vertex_mask(self.highlighted_translators)
            new_translator_axis_colors = attenuation_factor * self.vertex_colors
            new_translator_axis_colors[highlight_mask] = self.vertex_colors[highlight_mask]

            new_translator_handle_colors = attenuation_factor * self.handle_colors
            new_translator_handle_colors[self.highlighted_translators] = self.handle_colors[self.highlighted_translators]

        return new_translator_axis_colors, new_translator_handle_colors

    @property
    def n_translators(self) -> int:
        """"The number of translators.

        Returns
        -------
        n_translators : int
            The number of translators.
        """
        return len(self.normal_vectors)

    @property
    def translator_indices(self) -> np.ndarray:
        """The translator index for each vertex.

        Returns
        -------
        translator_indices : np.ndarray
            (n_translators * 2,) array containing the rotator index for each vertex.
        """
        return self._translator_indices

    def translator_vertex_mask(self, translator_indices: Union[List[int], int]) -> np.ndarray:
        """Create a boolean mask to select the vertices from the specified translators.

        Parameters
        ----------
        translator_indices : Union[List[int], int]
            The indices of the translators to set to True in the resulting mask.

        Returns
        -------
        vertex_mask : np.ndarray
            The (n_vertices,) array containing True values where the vertices are
            from the selected rotator(s).
        """
        if isinstance(translator_indices, int):
            translator_indices = [translator_indices]

        return np.isin(self.translator_indices, translator_indices)


class RotatorModel:
    def __init__(
            self,
            normal_vectors: Union[np.ndarray, List[int]],
            colors: np.ndarray,
            radius: float = 20,
            n_segments: int = 64,
            width: float = 5,
            handle_size: float = 3
    ):
        self._normal_vector_indices = normal_vectors
        self.radius = radius
        self.n_segments = n_segments
        self.width = width
        self.handle_size = handle_size

        vertices, connections, vertex_colors, handle_points, handle_colors, rotator_indices = make_rotator_data(
            rotator_normals=self.normal_vectors,
            rotator_colors=colors,
            center_point=np.array([0, 0, 0]),
            radius=self.radius,
            n_segments=self.n_segments
        )

        self._vertices = vertices
        self._connections = connections
        self._vertex_colors = vertex_colors
        self._handle_points = handle_points
        self._handle_colors = handle_colors
        self._rotator_indices = rotator_indices

        # array
        self._highlighted_rotators: Optional[List[int]] = None

    @property
    def normal_vectors(self) -> np.ndarray:
        """Normal vectors for the rotation planes supported by this manipulator.

        Returns
        -------
        normal_vectors : np.ndarray
            The normal vectors in data coordinates for the untransformed manipulator
        """
        return MANIPULATOR_BASIS[self._normal_vector_indices]

    @property
    def vertices(self) -> np.ndarray:
        """Coordinates for the rotator arc vertices.

        Returns
        -------
        vertices : np.ndarray
            (n_rotators * n_segments, 3) array containing the coordinates
            of all vertices.
        """
        return self._vertices

    @property
    def connections(self) -> np.ndarray:
        """Connections between vertices in the rotator arc.

        Returns
        -------
        connections : np.ndarray
            (n_rotators * [n_segments - 1], 2) array containing the
            connections between arc vertices.
        """
        return self._connections

    @property
    def vertex_colors(self) -> np.ndarray:
        """The color for each vertex.

        Returns
        -------
        vertex_colors : np.ndarray
            (n_rotators * n_segments, 4) array containing RGBA colors
            for all vertices.
        """
        return self._vertex_colors


    @property
    def handle_points(self) -> np.ndarray:
        """The coordinates for each handle point.

        Returns
        -------
        handle_points : np.ndarray
            (n_rotators, 3) array containing the coordinates of the handle for
            each rotator.
        """
        return self._handle_points

    @property
    def handle_colors(self) -> np.ndarray:
        """The color for each rotator handle.

        Returns
        handle_colors : np.ndarray
            (n_rotators, 4) RGBA array containing the color for each translator handle.
        """
        return self._handle_colors

    @property
    def highlighted_rotators(self) -> Optional[List[int]]:
        """Which, if any, rotators are highlighted.

        Returns
        -------
        highlighted_translators
            The rotators that are highlighted.
            Returns None if no rotators are highlighted.
            Returns a list with length == 0 if all rotators are attenuated.
            Returns a list with length > 0 if some rotators are highlighted.
            The list contains the indices for each highlighted rotator.
        """
        return self._highlighted_rotators

    @highlighted_rotators.setter
    def highlighted_rotators(self, highlighted_rotators: Optional[List[int]]) -> None:
        if highlighted_rotators is None:
            self._highlighted_rotators = highlighted_rotators
        else:
            self._highlighted_rotators = list(highlighted_rotators)

    @property
    def rendered_rotator_colors(self) -> Tuple[np.ndarray, np.ndarray]:
        """"The colors for the rotator as they should be rendered taking highlighting into account.

        Returns
        -------
        new_rotator_arc_colors : np.ndarray
            (n_vertices, 4) RGBA array containing the color for each vertex in the rotators arcs.
        new_rotator_handle_colors : np.ndarray
            (n_handles, 4) RGBA array containing the color for each rotator handle.
        """
        attenuation_factor = 0.5
        if self.highlighted_rotators is None:
            new_rotator_arc_colors = self.vertex_colors
            new_rotator_handle_colors = self.handle_colors
        else:
            highlight_mask = self.rotator_vertex_mask(self.highlighted_rotators)
            new_rotator_arc_colors = attenuation_factor * self.vertex_colors
            new_rotator_arc_colors[highlight_mask] = self.vertex_colors[highlight_mask]

            new_rotator_handle_colors = attenuation_factor * self.handle_colors
            new_rotator_handle_colors[self.highlighted_rotators] = self.handle_colors[self.highlighted_rotators]

        return new_rotator_arc_colors, new_rotator_handle_colors

    @property
    def n_rotators(self) -> int:
        """"The number of rotators.

        Returns
        -------
        n_translators : int
            The number of translators.
        """
        return len(self.normal_vectors)

    @property
    def rotator_indices(self) -> np.ndarray:
        """The rotator index for each vertex.

        Returns
        -------
        rotator_indices : np.ndarray
            (n_rotators * n_segments,) array containing the rotator index for each vertex.
        """
        return self._rotator_indices

    def rotator_vertex_mask(self, rotator_indices: Union[List[int], int]) -> np.ndarray:
        """Create a boolean mask to select the vertices from the specified rotators.

        Parameters
        ----------
        rotator_indices : Union[List[int], int]
            The indices of the rotators to set to True in the resulting mask.

        Returns
        -------
        vertex_mask : np.ndarray
            The (n_vertices,) array containing True values where the vertices are
            from the selected rotator(s).
        """
        if isinstance(rotator_indices, int):
            rotator_indices = [rotator_indices]

        return np.isin(self.rotator_indices, rotator_indices)
