from abc import ABC, abstractmethod
from typing import Type

import napari
from napari.utils.events import Event
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout

from .utils.napari_utils import generate_populated_layer_selection_widget


class ThreeDeeModel(ABC):
    """Base class for manipulators and annotators.

    To implement:
        - the __init__() should take the viewer as the first argument and all
        keyword arguments should have default values.
        - implement the set_layers() method
        - implement the _on_enable() callback
        - implement the _on_disable() callback
    """

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._on_enable() if value is True else self._on_disable()
        self._enabled = value

    @abstractmethod
    def set_layers(self, *args):
        """This method should set layer attributes on the manipulator/annotator.
        Arguments to this function should be typed as napari layers.
        """
        pass

    @abstractmethod
    def _on_enable(self):
        """This method should 'activate' the manipulator/annotator,
        setting state and connecting callbacks.
        """
        pass

    @abstractmethod
    def _on_disable(self):
        """This method should 'deactivate' the manipulator/annotator,
        updating state and disconnecting callbacks.
        """
        pass


class QtThreeDeeWidgetBase(QWidget):
    """Base class for GUI elements in napari-threedee."""

    def __init__(self, model: Type[ThreeDeeModel], viewer: napari.Viewer, flags=None, *args,
                 **kwargs):
        super().__init__(flags, *args, **kwargs)
        self.model = model(viewer)
        self.viewer = viewer

        self._layer_selection_widget = generate_populated_layer_selection_widget(
            func=self.model.set_layers, viewer=viewer
        )
        self._layer_selection_widget()

        # start in the disabled state
        self.model.enabled = False
        self.activate_button = QPushButton('activate')
        self.activate_button.setCheckable(True)
        self.activate_button.setChecked(False)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.layer_selection_widget)
        self.layout().addWidget(self.activate_button)

        self.activate_button.clicked.connect(self.on_activate_button_click)

    @property
    def layer_selection_widget(self) -> QWidget:
        return self._layer_selection_widget.native

    def on_activate_button_click(self, event: Event):
        if self.activate_button.isChecked() is True:
            self.model.enabled = True
            self.activate_button.setText('deactivate')
        else:
            self.model.enabled = False
            self.activate_button.setText('activate')
