from enum import Enum
from typing import Optional, Tuple, Union

import napari
from napari.layers import Image
from napari.utils.events.event import EmitterGroup, Event
from napari.utils.geometry import rotation_matrix_from_vectors_3d
import numpy as np

from .._backend.threedee_model import ThreeDeeModel
from ..annotators.spline_annotator import SplineAnnotator


class CameraSplineMode(Enum):
    PAN_ZOOM = "pan_zoom"
    EXPLORE = "explore"
    ANNOTATE = "annotate"


class CameraSpline(ThreeDeeModel):
    """Model for a spline that is used to direct the camera path."""
    COLOR_CYCLE = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
    ]
    SPLINE_ID_COLUMN: str = "spline_id"
    N_SPLINE_POINTS = 1000

    def __init__(
            self,
            viewer: napari.Viewer,
            image_layer: Optional[Image] = None,
            enabled: bool = False
    ):

        self.events = EmitterGroup(
            source=self,
            spline_valid=Event,
            mode=Event
        )

        self.viewer = viewer
        self._image_layer = image_layer
        self._mode = CameraSplineMode("pan_zoom")
        self._spline_valid = False
        self._up_direction = (1, 0, 0)
        self._view_direction_transformation = np.eye(3)
        self._current_spline_coordinate = 0

        self.spline_annotator_model = SplineAnnotator(viewer=viewer, image_layer=None, enabled=False)
        self.spline_annotator_model.events.splines_updated.connect(self._check_if_spline_valid)

        self._check_if_spline_valid()
        self.enabled = enabled

    @property
    def mode(self) -> CameraSplineMode:
        """The current mode for interaction.

        Returns
        -------
        mode : CameraSplineMode
            PAN_ZOOM: normal napari pan/zooming interaction
            ANNOTATE:
            EXPLORE: The camera follows the spline.
                The position is set by CameraSpline.current_spline_coordinate.
        """
        return self._mode

    @mode.setter
    def mode(self, mode: Union[str, CameraSplineMode]):
        """The current mode for interaction.

        Parameters
        -------
        mode : Union[str, CameraSplineMode]
            PAN_ZOOM: normal napari pan/zooming interaction
            ANNOTATE:
            EXPLORE: The camera follows the spline.
                The position is set by CameraSpline.current_spline_coordinate.
        """
        if isinstance(mode, str):
            mode = CameraSplineMode(mode.lower())
        if mode == self.mode:
            # don't do anything if the mode is unchanged
            return
        if mode == CameraSplineMode.PAN_ZOOM:
            self.stop_spline_annotation()
            self.stop_spline_exploration()
        elif mode == CameraSplineMode.EXPLORE:
            self.stop_spline_annotation()
            self.start_spline_exploration()
        elif mode == CameraSplineMode.ANNOTATE:
            self.stop_spline_exploration()
            self.start_spline_annotation()
        else:
            raise ValueError("invalid mode")
        self._mode = mode
        self.events.mode()

    @property
    def spline_valid(self) -> bool:
        """Flag set to True when a valid spline has been annotated"""
        return self._spline_valid

    @spline_valid.setter
    def spline_valid(self, valid: bool):
        """Flag set to True when a valid spline has been annotated"""
        if valid == self._spline_valid:
            # if the value is unchanged, do nothing
            return
        self._spline_valid = valid
        self.events.spline_valid()

    @property
    def view_direction_transformation(self) -> np.ndarray:
        """The transformation to apply to the view direction when viewing along the spline.

        Returns
        -------
        view_direction_transformation : np.ndarray
            (3, 3) array containing the transformation
            to be applied to the view direction when
            sliding along the spline
        """
        return self._view_direction_transformation

    @view_direction_transformation.setter
    def view_direction_transformation(self, transformation: np.ndarray):
        """The transformation to apply to the view direction when viewing along the spline.

        Returns
        -------
        view_direction_transformation : np.ndarray
            (3, 3) array containing the transformation
            to be applied to the view direction when
            sliding along the spline.
        """
        self._view_direction_transformation = transformation

    @property
    def up_direction(self) -> Tuple[float, float, float]:
        """The 3D vector for the camera up direction when sliding along the spline.

        Returns
        -------
        up_direction : Tuple[float, float, float]
            The up direction in 3D displayed data units.
        """
        return self._up_direction

    @up_direction.setter
    def up_direction(self, up_vector: Tuple[float, float, float]):
        """The 3D vector for the camera up direction when sliding along the spline.

        Parameters
        ----------
        up_direction : Tuple[float, float, float]
            The up direction in 3D displayed data units.
        """
        self._up_direction = up_vector

    @property
    def current_spline_coordinate(self) -> float:
        """The current coordinate to view the spline at.

        Returns
        -------
        current_spline_coordinate : float
            The current spline coordinate. Goes from 0 to 1.
        """
        return self._current_spline_coordinate

    @current_spline_coordinate.setter
    def current_spline_coordinate(self, current_spline_coordinate: float):
        self._current_spline_coordinate = current_spline_coordinate
        self.set_camera_position(self.current_spline_coordinate)

    def calculate_transform_from_spline_tangent_to_view_direction(self):
        current_view_direction = self.viewer.camera.view_direction

        spline_dict = self.spline_annotator_model.points_layer.metadata["splines"]

        # only one spline
        spline_object = spline_dict[0]
        spline_tangent = np.squeeze(spline_object._sample_backbone(u=[self.current_spline_coordinate], derivative=1))
        spline_tangent_displayed = spline_tangent[list(self.viewer.dims.displayed)]

        self.view_direction_transformation = rotation_matrix_from_vectors_3d(
            spline_tangent_displayed, current_view_direction
        )

        # get the current up direction
        self.up_direction = self.viewer.camera.up_direction

    def _check_if_spline_valid(self, event=None):
        """Check if there is a valid spline to slice along"""
        spline_points_layer = self.spline_annotator_model.points_layer
        if spline_points_layer is None:
            self.spline_valid = False
            return

        spline_dict = spline_points_layer.metadata["splines"]
        spline_object = spline_dict.get(0, None)
        if spline_object is None:
            self.spline_valid = False
        else:
            self.spline_valid = True

    @property
    def image_layer(self) -> Image:
        """The image layer to set the spline points on.

        The visualization plane for the image layer is used to set
        the plane the spline points are added on.
        """
        return self._image_layer

    @image_layer.setter
    def image_layer(self, layer: Image):
        self._image_layer = layer

    def _on_enable(self):
        """Function called when the widget is activated"""
        self.spline_annotator_model.set_layers(self.image_layer)

    def _on_disable(self):
        """Function called when the widget is deactivated"""
        self.spline_annotator_model.enabled = False
        self.mode = CameraSplineMode("pan_zoom")

    def set_layers(self, image_layer: napari.layers.Image):
        self.image_layer = image_layer

    def set_camera_position(self, spline_coordinate: float):
        """Set the viewer camera position along the spline.

        Parameters
        ----------
        spline_coordinate : float
            The position along the spline the set the viewer
            camera. Should be in the spline coordinate system
            from 0 to 1.
        """
        if self.image_layer is None:
            # do not do anything if the image layer hasn't been set
            return

        if self.spline_valid is False:
            # do not do anything if there isn't a valid spline
            return

        spline_dict = self.spline_annotator_model.points_layer.metadata["splines"]

        # only one spline
        spline_object = spline_dict[0]
        spline_point = spline_object._sample_backbone(u=[spline_coordinate])
        self.viewer.camera.center = np.squeeze(spline_point)

        view_direction = np.squeeze(spline_object._sample_backbone(u=[spline_coordinate], derivative=1))
        view_direction_displayed = view_direction[list(self.viewer.dims.displayed)]

        view_direction_transformed = tuple(self._transform_view_direction(view_direction_displayed))

        self.viewer.camera.set_view_direction(
            view_direction=view_direction_transformed, up_direction=self.up_direction
        )

    def _transform_view_direction(self, view_direction: np.ndarray) -> np.ndarray:
        """Transform the view direction along the spline to the view set by the user.

        The transformation applied is in self.view_direction_transformation.

        Parameters
        ----------
        view_direction : np.ndarray
            The view direction in displayed data coordinates.
            Should be a (3,) array.

        Returns
        -------
        transformed_view_direction : np.ndarray
            The view direction transformed by self.view_direction_transformation
        """
        return self.view_direction_transformation.dot(view_direction)

    def start_spline_annotation(self):
        """Callback called when entering ANNOTATION mode."""
        self.spline_annotator_model.enabled = True

        # disable the key binding to switch to the next spline index
        self.viewer.bind_key('n', None)

    def stop_spline_annotation(self):
        """Callback called when exiting ANNOTATION mode."""
        self.spline_annotator_model.enabled = False

    def start_spline_exploration(self):
        """Setup up the exploration mode.

        This is called when mode is switched to exploration mode.
        """
        self.set_camera_position(self.current_spline_coordinate)

    def stop_spline_exploration(self):
        """Clean up the exploration mode.

        This is called when mode is switched from exploration mode.
        """
        # currently nothing is done - added for completeness
        pass

