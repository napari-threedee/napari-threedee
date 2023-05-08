import napari
from napari.utils.events import Event
from qtpy.QtWidgets import QPushButton, QGroupBox, QVBoxLayout, QSpinBox, QLabel, QHBoxLayout

from napari_threedee._backend.threedee_widget_base import QtThreeDeeWidgetBase

from napari_threedee.annotators.surfaces.annotator import SurfaceAnnotator



class QtSurfaceAnnotatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(model_class=SurfaceAnnotator, viewer=viewer)

        surface_layout = QHBoxLayout()
        self.active_surface_spinbox = QSpinBox()
        self.active_surface_spinbox.setMinimum(0)
        self.active_surface_spinbox.setValue(self.model.active_surface_id)
        self.active_surface_spinbox.valueChanged.connect(
            self._on_current_surface_changed
        )

        surface_layout.addWidget(QLabel("surface:"))
        surface_layout.addWidget(self.active_surface_spinbox)

        level_layout = QHBoxLayout()
        self.active_level_spinbox = QSpinBox()
        self.active_level_spinbox.setMinimum(0)
        self.active_level_spinbox.setValue(self.model.active_level_id)
        self.active_level_spinbox.valueChanged.connect(
            self._on_current_level_changed
        )

        level_layout.addWidget(QLabel("level:"))
        level_layout.addWidget(self.active_level_spinbox)

        # add the fitting UI to the group_box
        self.update_surface_button = QPushButton("fit surface")
        self.update_surface_button.pressed.connect(self._draw_surface)

        self.fitting_group_box = QGroupBox("fit surface")
        group_box_layout = QVBoxLayout()
        group_box_layout.addLayout(surface_layout)
        group_box_layout.addLayout(level_layout)
        group_box_layout.addWidget(self.update_surface_button)
        self.fitting_group_box.setLayout(group_box_layout)
        self.fitting_group_box.setVisible(False)

        # add the widget to the layout
        self.layout().addWidget(self.fitting_group_box)

        # connect events to sync changes in the model to the UI
        self.model.events.active_surface_id.connect(self._update_active_surface)
        self.model.events.active_level_id.connect(self._update_active_level)

    def _draw_surface(self):
        self.model._draw_splines()
        self.model._draw_surface()

    def _on_current_surface_changed(self, event: Event):
        self.model.active_surface_id = self.active_surface_spinbox.value()

    def _on_current_level_changed(self, event: Event):
        self.model.active_spline_id = self.active_level_spinbox.value()

    def on_activate_button_click(self, event: Event):
        """Callback function when the activate button is clicked"""
        if self.activate_button.isChecked() is True:
            self.model.enabled = True
            self.activate_button.setText('deactivate')
            self.fitting_group_box.setVisible(True)
        else:
            self.model.enabled = False
            self.activate_button.setText('activate')
            self.fitting_group_box.setVisible(False)

    def _update_active_surface(self):
        """Update surface spinbox when surface id has changed on the model."""
        self.active_surface_spinbox.setValue(self.model.active_surface_id)

    def _update_active_level(self):
        """Update the spline id spinbox when the model has changed value."""
        self.active_level_spinbox.setValue(self.model.active_level_id)


