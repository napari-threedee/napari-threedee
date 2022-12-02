from typing import Type

import napari
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout
from napari.utils.events import Event

from napari_threedee._backend.threedee_model import ThreeDeeModel
from ..utils.napari_utils import generate_populated_layer_selection_widget


class QtThreeDeeWidgetBase(QWidget):
    """Base class for GUI elements in napari-threedee."""

    def __init__(self, model_class: Type[ThreeDeeModel], viewer: napari.Viewer, flags=None, *args,
                 **kwargs):
        super().__init__(flags, *args, **kwargs)
        self.model: ThreeDeeModel = model_class(viewer)
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
