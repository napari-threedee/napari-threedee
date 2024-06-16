from typing import Type

import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QWidget, QPushButton, QVBoxLayout
from napari.utils.events import Event
from napari.utils.notifications import show_info

from napari_threedee._backend.threedee_model import N3dComponent
from ..utils.napari_utils import generate_populated_layer_selection_widget


class QtThreeDeeWidgetBase(QWidget):
    """Base class for GUI elements in napari-threedee."""

    def __init__(self, model_class: Type[N3dComponent], viewer: napari.Viewer, flags=None, *args,
                 **kwargs):
        super().__init__(flags, *args, **kwargs)

        self.viewer = viewer
        self._model_class = model_class

        self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_change)
        self.setLayout(QVBoxLayout())
        
        if self.viewer.dims.ndim<3:
            show_info("Only 3D+ data is supported, widget is deactivated.")
            self.ndim_info = QLabel("2D data, deactivated.")
            self.layout().addWidget(self.ndim_info)
        elif self.viewer.dims.ndisplay == 2:
            show_info("Viewer needs to be in 3D mode, widget is deactivated.")
            self.ndim_info = QLabel("2D display, deactivated\nSwitch to 3D mode to activate")
            self.layout().addWidget(self.ndim_info)
        else:
            self.initialize_widget(flags=None)


    def initialize_widget(self, flags=None):
        self.model: N3dComponent = self._model_class(self.viewer)
        
        self._layer_selection_widget = generate_populated_layer_selection_widget(
            func=self.model.set_layers, viewer=self.viewer
        )
        self._layer_selection_widget()

        # start in the disabled state
        self.model.enabled = False
        self.activate_button = QPushButton('activate')
        self.activate_button.setCheckable(True)
        self.activate_button.setChecked(False)

        self.layout().insertWidget(0, self.layer_selection_widget)
        self.layout().insertWidget(1, self.activate_button)
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

    def _on_ndisplay_change(self, event):
        new_ndisplay = event.value
        if self.viewer.dims.ndim < 3:
            return
        if new_ndisplay == 2:
            self.activate_button.setEnabled(False) 
            self.activate_button.setText('2D, disabled')
        elif new_ndisplay==3 and not hasattr(self, "model"):
            self.ndim_info.hide()
            self.layout().removeWidget(self.ndim_info)
            self.ndim_info.deleteLater()
            self.layout().update()
            self.initialize_widget(flags=None)
            self.layout().update()
        else:
            self.activate_button.setEnabled(True)
            self.model.enabled = True
            self.activate_button.setChecked(True)
            self.activate_button.setText('deactivate')
