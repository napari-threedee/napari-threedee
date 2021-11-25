import numpy as np
from napari.utils.geometry import project_points_onto_plane, rotation_matrix_from_vectors
from napari_threedee.manipulators.base_manipulator import BaseManipulator
from napari_threedee.manipulators.manipulator_utils import create_circle_line_segments, make_rotator_meshes, make_translator_meshes


class RenderPlaneManipulator(BaseManipulator):

    def __init__(self, viewer, layer, line_length=50, rotator_radius=5, order=0):
        self._line_length = line_length
        self._rotator_radius = rotator_radius
        self._initial_translate = None
        self._initial_click_vector = None
        self._initial_plane_pos = None

        self._rotator_angle_offset = 0
        self._initial_rotator_normals = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ]
        )

        super().__init__(viewer, layer, order)

    @property
    def line_length(self):
        return self._line_length

    @line_length.setter
    def line_length(self, line_length):
        self._line_length = line_length
        self._on_data_change()

    @property
    def rotator_radius(self):
        return self._rotator_radius

    @line_length.setter
    def line_length(self, rotator_radius):
        self._rotator_radius = rotator_radius
        self._on_data_change()


    def _init_arrow_lines(self):
        # note order is x, y, z for VisPy
        self.centroid = self._layer.experimental_slicing_plane.position
        normal = self._layer.experimental_slicing_plane.normal
        self._initial_translator_normals = np.asarray([normal])

        translator_vertices, translator_indices, translator_colors, triangle_indices = make_translator_meshes(
            centroids=np.asarray([0, 0, 0]),
            normals=self._initial_translator_normals,
            colors=self._default_color[:len(self._initial_translator_normals)],
            translator_length=self.line_length,
            tube_radius=0.3,
            tube_points=3,
        )

        self.translator_vertices = translator_vertices
        self.translator_indices = translator_indices
        self.translator_colors = translator_colors
        self.translator_triangle_indices = triangle_indices

        self.translator_normals = self._initial_translator_normals.copy()

    def _init_rotators(self):
        self._initial_rotator_normals[0] = self._layer.experimental_slicing_plane.normal

        rotator_vertices, rotator_indices, rotator_colors, triangle_indices = make_rotator_meshes(
            centroids=np.repeat([0, 0, 0], 3, axis=0),
            normals=self._initial_rotator_normals,
            colors=self._default_color[:len(self._initial_rotator_normals)],
            rotator_radius=self.rotator_radius,
            tube_radius=0.3,
            tube_points=3,
            n_segments=self._N_SEGMENTS_ROTATOR
        )

        self.rotator_vertices = rotator_vertices
        self.rotator_indices = rotator_indices
        self.rotator_colors = rotator_colors
        self.rotator_triangle_indices = triangle_indices

        self.rotator_normals = self._initial_rotator_normals.copy()

    def _pre_drag(self, click_point_data_displayed, rotator_index):
        self._initial_plane_pos = self._layer.experimental_slicing_plane.position

        if rotator_index is not None:

            normal = self.rotator_normals[rotator_index]
            # project the initial click point onto the rotation plane
            centroid = self._layer.experimental_slicing_plane.position
            initial_click_point, _ = project_points_onto_plane(
                points=click_point_data_displayed,
                plane_point=centroid,
                plane_normal=normal,
            )

            self._initial_click_vector = np.squeeze(initial_click_point) - centroid
            self._pre_drag_rotator_normals = self.rotator_normals.copy()
            self._pre_drag_translator_normals = self.translator_normals.copy()
            self._initial_rot_mat = self.rot_mat.copy()

    def _while_translator_drag(self, translation_vector: np.ndarray):
        if translation_vector is not None:
            new_translation = self._initial_plane_pos + translation_vector
            self._layer.experimental_slicing_plane.position = np.squeeze(new_translation)
            self.centroid = self._layer.experimental_slicing_plane.position
        else:
            pass

        self._on_matrix_change()
        # self._on_data_change()

    def _while_rotator_drag(self, click_position, rotation_drag_vector, rotator_index):
        if rotator_index == 0:
            centroid = self._layer.experimental_slicing_plane.position
            normal = self.rotator_normals[0]
            projected_click_point, _ = project_points_onto_plane(
                points=click_position,
                plane_point=centroid,
                plane_normal=normal,
            )
            click_vector = np.squeeze(projected_click_point) - centroid

            rot_mat = rotation_matrix = rotation_matrix_from_vectors(
                self._initial_click_vector, click_vector
            )

            self.rot_mat = np.dot(rot_mat, self._initial_rot_mat)
            self.rotator_normals = self._pre_drag_rotator_normals @ rot_mat.T
            self._on_matrix_change()

        elif (rotator_index == 1) or (rotator_index == 2):
            self._rotate_plane(
                click_position=click_position,
                rotator_index=rotator_index
            )

    def _rotate_plane(self, click_position, rotator_index):
        plane_normal = self.rotator_normals[rotator_index]
        print(plane_normal)
        centroid = self._layer.experimental_slicing_plane.position
        projected_click_point, _ = project_points_onto_plane(
            points=click_position,
            plane_point=centroid,
            plane_normal=plane_normal,
        )
        click_vector = np.squeeze(projected_click_point) - centroid
        rot_mat = rotation_matrix = rotation_matrix_from_vectors(
            self._initial_click_vector, click_vector
        )
        self.rotator_normals = self._pre_drag_rotator_normals @ rot_mat.T
        self.translator_normals = self._pre_drag_translator_normals @ rot_mat.T
        self._layer.experimental_slicing_plane.normal = self.rotator_normals[0]

        self.rot_mat = np.dot(rot_mat, self._initial_rot_mat)
        self._on_matrix_change()


    def _on_click_cleanup(self):
        self._initial_plane_pos = None
        self._initial_click_vector = None
        self._initial_rot_mat = None
