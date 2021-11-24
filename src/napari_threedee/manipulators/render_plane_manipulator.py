import numpy as np
from napari.utils.geometry import project_points_onto_plane, rotation_matrix_from_vectors
from napari_threedee.manipulators.base_manipulator import BaseManipulator
from napari_threedee.manipulators.manipulator_utils import create_circle_line_segments


class RenderPlaneManipulator(BaseManipulator):

    def __init__(self, viewer, layer, line_length=50, rotator_radius=5, order=0):
        self._line_length = line_length
        self._rotator_radius = rotator_radius
        self._initial_translate = None
        self._initial_click_vector = None
        self._initial_plane_pos = None
        self._initial_ortho_rotator_1_normal = None
        self._initial_ortho_rotator_2_normal = None

        self._rotator_angle_offset = 0
        self.ortho_rotator_1_normal = np.array([0, 1, 0])
        self.ortho_rotator_2_normal = np.array([0, 0, 1])


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
        centroid = self._layer.experimental_slicing_plane.position
        normal = self._layer.experimental_slicing_plane.normal
        self._line_data2D = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]]
        )
        line_data3D = self.line_length * np.array(
            [[0, 0, 0], normal]
        )
        self._line_data3D = line_data3D + centroid

    def _init_rotators(self):
        centroid = self._layer.experimental_slicing_plane.position
        normal = self._layer.experimental_slicing_plane.normal

        self._in_plane_rotator = create_circle_line_segments(
            centroid=centroid,
            normal=normal,
            radius=self.rotator_radius,
            n_segments=self._N_SEGMENTS_ROTATOR
        )

        ortho_rotator_0 = create_circle_line_segments(
            centroid=centroid,
            normal=self.ortho_rotator_1_normal,
            radius=self.rotator_radius,
            n_segments=self._N_SEGMENTS_ROTATOR
        )

        ortho_rotator_1 = create_circle_line_segments(
            centroid=centroid,
            normal=self.ortho_rotator_2_normal,
            radius=self.rotator_radius,
            n_segments=self._N_SEGMENTS_ROTATOR
        )

        self._rotator_data3D = np.concatenate([self._in_plane_rotator, ortho_rotator_0, ortho_rotator_1])

        in_plane_rotator_connections = np.column_stack(
            [np.arange(self._N_SEGMENTS_ROTATOR - 1), np.arange(1, self._N_SEGMENTS_ROTATOR)]
        )
        ortho_rotator_0_connections = in_plane_rotator_connections + self._N_SEGMENTS_ROTATOR
        ortho_rotator_1_connections = in_plane_rotator_connections + (2 * self._N_SEGMENTS_ROTATOR)

        self._rotator_connections = np.concatenate(
            [in_plane_rotator_connections, ortho_rotator_0_connections, ortho_rotator_1_connections])

        # each value is the axis a given vertex corresponds to
        self._rotator_vertex_axis = np.repeat([0, 1, 2], self._N_SEGMENTS_ROTATOR)

    def _pre_drag(self, click_point_data_displayed, rotator_index):
        self._initial_plane_pos = self._layer.experimental_slicing_plane.position

        if rotator_index is not None:
            if rotator_index == 0:
                normal = self._layer.experimental_slicing_plane.normal
            elif (rotator_index == 1) or (rotator_index == 2):
                normal = getattr(self, f'ortho_rotator_{rotator_index}_normal')
            # project the initial click point onto the rotation plane
            centroid = self._layer.experimental_slicing_plane.position
            initial_click_point, _ = project_points_onto_plane(
                points=click_point_data_displayed,
                plane_point=centroid,
                plane_normal=normal,
            )

            self._initial_click_vector = np.squeeze(initial_click_point) - centroid
            self._initial_normal = self._layer.experimental_slicing_plane.normal
            self._initial_ortho_rotator_1_normal = self.ortho_rotator_1_normal
            self._initial_ortho_rotator_2_normal = self.ortho_rotator_2_normal

    def _while_translator_drag(self, translation_vector: np.ndarray, rotator_drag_vector: np.ndarray, rotator_selection: np.ndarray):
        if translation_vector is not None:
            new_translation = self._initial_plane_pos + translation_vector
            self._layer.experimental_slicing_plane.position = np.squeeze(new_translation)
        else:
            pass

        self._init_arrow_lines()
        self._init_rotators()
        self._on_data_change()

    def _while_rotator_drag(self, click_position, rotation_drag_vector, rotator_index):
        if rotator_index == 0:
            centroid = self._layer.experimental_slicing_plane.position
            normal = self._layer.experimental_slicing_plane.normal
            projected_click_point, _ = project_points_onto_plane(
                points=click_position,
                plane_point=centroid,
                plane_normal=normal,
            )
            click_vector = np.squeeze(projected_click_point) - centroid

            rot_mat = rotation_matrix = rotation_matrix_from_vectors(
                self._initial_click_vector, click_vector
            )

            self.ortho_rotator_1_normal = np.dot(rot_mat, self._initial_ortho_rotator_1_normal)
            self.ortho_rotator_2_normal = np.dot(rot_mat, self._initial_ortho_rotator_2_normal)

            self._init_rotators()
            self._on_data_change()
        elif (rotator_index == 1) or (rotator_index == 2):
            self._rotate_plane(
                click_position=click_position,
                rotator_index=rotator_index
            )

    def _rotate_plane(self, click_position, rotator_index):
        plane_normal = getattr(self, f'ortho_rotator_{rotator_index}_normal')
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
        self._layer.experimental_slicing_plane.normal = np.dot(rot_mat, self._initial_normal)

        self.ortho_rotator_1_normal = np.dot(rot_mat, self._initial_ortho_rotator_1_normal)
        self.ortho_rotator_2_normal = np.dot(rot_mat, self._initial_ortho_rotator_2_normal)

        self._init_arrow_lines()
        self._init_rotators()
        self._on_data_change()


    def _on_click_cleanup(self):
        self._initial_plane_pos = None
        self._initial_click_vector = None
        self._initial_ortho_rotator_1_normal = None
        self._initial_ortho_rotator_2_normal = None
