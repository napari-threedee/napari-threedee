from typing import Type

import napari
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout
from napari.utils.events import Event
from napari.utils.notifications import show_info

from napari_threedee._backend.threedee_model import N3dComponent
from ..utils.napari_utils import generate_populated_layer_selection_widget


class QtThreeDeeWidgetBase(QWidget):
    """Base class for GUI elements in napari-threedee."""

    def __init__(self, model_class: Type[N3dComponent], viewer: napari.Viewer, flags=None, *args,
                 **kwargs):
        super().__init__(flags, *args, **kwargs)
        # initialize the model disabled
        self.model: N3dComponent = model_class(viewer, enabled=False)
        self.viewer = viewer

        self._layer_selection_widget = generate_populated_layer_selection_widget(
            func=self.model.set_layers, viewer=viewer
        )
        self._layer_selection_widget()

        # start in the disabled state
        self.activate_button = QPushButton('activate')
        self.activate_button.setCheckable(True)
        self.activate_button.setChecked(False)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.layer_selection_widget)
        self.layout().addWidget(self.activate_button)

        # Don't allow activating in 2D mode or with <3D data
        if self.viewer.dims.ndisplay == 2 or self.viewer.dims.ndim < 3:
            show_info("Viewer needs to be in 3D mode with 3D+ data.")
            self.activate_button.setEnabled(False)
            self.activate_button.setText('2D data/viewer, disabled')
        
        self.activate_button.clicked.connect(self.on_activate_button_click)
        self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_change)

    @property
    def layer_selection_widget(self) -> QWidget:
        return self._layer_selection_widget.native

    def on_activate_button_click(self, event: Event):
        if self.activate_button.isChecked() is True:
            self.model.enabled = True
            self.activate_button.setText('deactivate')
        else:
            self.model.enabled = False
            # permit deactivating in 2D mode, but don't permit reactivation
            if self.viewer.dims.ndisplay == 2 or self.viewer.dims.ndim < 3:
                self.activate_button.setEnabled(False)
                self.activate_button.setText('2D data/viewer, disabled')
            else:
                self.activate_button.setText('activate')

    def _on_ndisplay_change(self, event):
        new_ndisplay = event.value
        # going to 2D and was never activated
        if (new_ndisplay == 2 or self.viewer.dims.ndim < 3) and self.activate_button.text() != 'deactivate':
            self.activate_button.setEnabled(False)
            self.activate_button.setText('2D data/viewer, disabled')
        # going to 3D with 3D+ data
        elif self.viewer.dims.ndim > 2:
            # if never activated, enable activation
            if self.activate_button.text() == '2D data/viewer, disabled':
                self.activate_button.setChecked(False)
                self.activate_button.setEnabled(True)
                self.activate_button.setText('activate')
            # otherwise assume was activated, so enable deactivation
            else:
                self.activate_button.setEnabled(True)
                self.activate_button.setChecked(True)
                self.activate_button.setText('deactivate')
