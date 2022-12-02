from abc import ABC, abstractmethod
from typing import Optional, Type

import napari
import numpy as np
from napari.viewer import Viewer

from napari_threedee._backend.threedee_model import ThreeDeeModel
from .._backend.manipulator.axis_model import AxisModel
from .._backend.manipulator.napari_manipulator_backend import \
    NapariManipulatorBackend
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, \
    remove_mouse_callback_safe


class BaseManipulator(ThreeDeeModel, ABC):
    """Base class for manipulator implementations.

    To implement:
        - the __init__() should take the viewer as the first argument, the layer
            as the second argument followed by any keyword arguments.
            Keyword arguments should have default values.
        - The __init__() should call the super.__init__().
        - implement a self._initialize_transform() method.
        - implement the _pre_drag() callback.
        - implement the _while_dragging_translator() callback.
        - implement the _while_dragging_rotator() callback.
        - implement the _post_drag() callback.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer containing the visuals.
    layer : Optional[Layer]
        The callback list of to attach the manipulator to.
    """

    def __init__(
        self,
        viewer,
        layer=None,
        rotator_axes: Optional[str] = None,
        translator_axes: Optional[str] = None,
        enabled: bool = True,
    ):
        super().__init__()
        self._viewer = viewer
        self._enabled = enabled
        self._backend = NapariManipulatorBackend(
            rotator_axes=rotator_axes,
            translator_axes=translator_axes,
            viewer=self._viewer,
            layer=layer,
        )
        self.visible = False
        self.layer = layer
        if self.enabled:
            self._on_enable()

    @property
    def origin(self) -> np.ndarray:
        """(3, ) array containing the origin of the manipulator."""
        return self._backend.manipulator_model.origin

    @origin.setter
    def origin(self, value: np.ndarray):
        self._backend.manipulator_model.origin = value

    @property
    def rotation_matrix(self) -> np.ndarray:
        """(3, 3) array containing the rotation matrix of the manipulator."""
        return self._backend.manipulator_model.rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, value: np.ndarray):
        self._backend.manipulator_model.rotation_matrix = value

    @property
    def z_vector(self) -> np.ndarray:
        return self.rotation_matrix[:, 0]

    @property
    def y_vector(self) -> np.ndarray:
        return self.rotation_matrix[:, 1]

    @property
    def x_vector(self) -> np.ndarray:
        return self.rotation_matrix[:, 2]

    @property
    def selected_translator(self) -> Optional[AxisModel]:
        if self._backend.manipulator_model.selected_object_type != 'translator':
            return None
        axis_id = self._backend.manipulator_model.selected_axis_id
        return AxisModel.from_id(axis_id)

    @property
    def selected_rotator(self) -> Optional[AxisModel]:
        if self._backend.manipulator_model.selected_object_type != 'rotator':
            return None
        axis_id = self._backend.manipulator_model.selected_axis_id
        return AxisModel.from_id(axis_id)

    @abstractmethod
    def _initialize_transform(self):
        """Update the origin and the rotation matrix of the manipulator visual.

        This should correctly initialize self.origin and self.rotation_matrix for the object
        you want to manipulate. This method will be called when the layer is updated.
        """
        ...

    def _pre_drag(self):
        """A callback called at the beginning of the drag event.

        This is typically used to save information that will be used during the
        translator/rotator drag callbacks (e.g. initial positions/orientations).
        """
        pass

    def _while_dragging_translator(self):
        """A callback called during translator drag events.

        Implementations should update the object being manipulated based on the state of the
        manipulator.
        """
        pass

    def _while_dragging_rotator(self):
        """A callback called during translator drag events.

        Implementations should update the object being manipulated based on the state of the
        manipulator.
        """
        pass

    def _post_drag(self):
        """A callback called at the end of the drag event.

        Implementations should use this to clean up any variables set during the click event.
        """
        pass

    def set_layers(self, layer: Type[napari.layers.Layer]):
        """Add the correct layer type for the manipulator in a subclass.

        Implementing this method with a correct type annotation will allow
        autogeneration of a Qt widget for a manipulator.
        """
        self.layer = layer

    def _connect_events(self):
        """This method should be implemented on subclasses that
        require events to be connected to the layer when self.layer
        is set (other than the main mouse callback"""
        pass

    def _disconnect_events(self):
        """This method must be implemented on subclasses that
        implement _connect_events(). This method is to disconnect
        the events that were connected in _connect_events()"""
        pass

    @property
    def layer(self):
        return self._backend.layer

    @layer.setter
    def layer(self, layer: Optional[Type[napari.layers.Layer]]):
        # if layer is None or layer == self.layer:
        #     return
        if layer is None:
            return
        self._disconnect_events()
        self._backend.layer = layer
        self._initialize_transform()
        if self.enabled:
            self._on_enable()
        else:
            self._on_disable()
        self._connect_events()

    @property
    def visible(self) -> bool:
        return self._backend.vispy_visual.visible

    @visible.setter
    def visible(self, value: bool):
        self._backend.vispy_visual.visible = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        self._on_enable() if self._enabled is True else self._on_disable()

    def _on_enable(self):
        if self.layer is not None:
            self.visible = True
            add_mouse_callback_safe(
                callback_list=self.layer.mouse_drag_callbacks,
                callback=self._mouse_callback,
                index=1
            )

    def _on_disable(self):
        if self.layer is not None:
            self.visible = False
            remove_mouse_callback_safe(
                self.layer.mouse_drag_callbacks,
                self._mouse_callback
            )

    def _on_ndisplay_change(self, event=None):
        if self._viewer.dims.ndisplay == 2:
            self.enabled = False
        else:
            self.enabled = True

    def _mouse_callback(self, layer, event):
        """Update the manipulated object via subclass implementations of drag/rotate behaviour."""
        yield
        if self._viewer.dims.ndisplay != 3 or self._backend.is_dragging is False:
            return  # early exit if manipulator is not being manipulated
        self._pre_drag()
        while event.type == 'mouse_move':
            selected_object_type = self._backend.manipulator_model.selected_object_type
            if selected_object_type == 'translator':
                self._while_dragging_translator()
            elif selected_object_type == 'rotator':
                self._while_dragging_rotator()
            yield
        self._post_drag()
