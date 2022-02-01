from typing import Optional

import numpy as np
from napari.utils.geometry import project_points_onto_plane, rotation_matrix_from_vectors_3d
from napari_threedee.manipulators.base_manipulator import BaseManipulator


class RenderPlaneManipulator(BaseManipulator):
    """A manipulator for moving an image layer rendering plane.

    Parameters
    ----------
    viewer : "napari.viewer.Viewer"
        The napari viewer containing the visuals.
    layer : Optional[napari.layers.base.Base]
        The layer to attach the manipulator to.
    order : int
        The order to place the manipulator visuals in the vispy scene graph.
    translator_length : float
        The length of the translator arms in data units.
    translator_width : float
        The width of the translator arms in data units.
    rotator_radius : float
        The radius of the rotators in data units.
    rotator_width : float
        The width of the rotators in data units.

    Attributes
    ----------
    centroid : np.ndarray
        (3, 1) array containing the coordinates to the centroid of the manipulator.
    rot_mat : np.ndarray
        (3, 3) array containing the rotation matrix applied to the manipluator.
    translator_length : float
        The length of the translator arms in data units.
    translator_width : float
        The width of the translator arms in data units.
    rotator_radius : float
        The radius of the rotators in data units.
    rotator_width : float
        The width of the rotators in data units.
    translator_normals : np.ndarray
        (N x 3) array containing the normal vector for each of the N translators.
    rotator_normals : np.ndarray
        (N x 3) array containing the normal vector for each of the N rotators.

    Notes
    -----
    _N_SEGMENTS_ROTATOR : float
        The number of segments to discretize the rotator into. More segments
        makes the rotator look more smooth, but will reduce rendering performance.
    _N_TUBE_POINTS : float
        The number of points to use to represent the circular crossection of the
        manipulator objects. More points makes the manipulator appear more smooth, but
        will reduce the rendering performance.
    """

    def __init__(self, viewer, layer, order=0, translator_length=50, rotator_radius=5):

        self._initial_plane_pos = None
        self._rotator_angle_offset = 0
        self._centroid = np.asarray(layer.experimental_slicing_plane.position)
        normal = np.asarray(layer.experimental_slicing_plane.normal)
        self._initial_translator_normals = np.asarray([normal])

        self._initial_rotator_normals = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ]
        )
        self._initial_rotator_normals[0] = layer.experimental_slicing_plane.normal

        super().__init__(
            viewer,
            layer,
            order=order,
            translator_length=translator_length,
            rotator_radius=rotator_radius
        )

    def _pre_drag(
            self,
            click_point: np.ndarray,
            selected_translator: Optional[int],
            selected_rotator: Optional[int]
    ):
        self._initial_plane_pos = np.asarray(self._layer.experimental_slicing_plane.position)
        self._initial_rot_mat = self.rot_mat.copy()

    def _while_translator_drag(self, selected_translator: int, translation_vector: np.ndarray):
        new_translation = self._initial_plane_pos + translation_vector
        self._layer.experimental_slicing_plane.position = np.squeeze(new_translation)
        self.centroid = np.asarray(self._layer.experimental_slicing_plane.position)

    def _while_rotator_drag(self, selected_rotator: int, rotation_matrix: np.ndarray):
        self._layer.experimental_slicing_plane.normal = self.rotator_normals[0]

    def _drag_callback_cleanup(self):
        self._initial_plane_pos = None
