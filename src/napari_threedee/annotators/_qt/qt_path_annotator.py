import napari
from napari.utils.events import Event
from qtpy.QtWidgets import QPushButton, QGroupBox, QVBoxLayout, QCheckBox, QSpinBox, QLabel, QHBoxLayout

from napari_threedee._backend.threedee_widget_base import QtThreeDeeWidgetBase

from napari_threedee.annotators.paths import PathAnnotator


class QtPathAnnotatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(model_class=PathAnnotator, viewer=viewer)

        self.auto_fit_checkbox = QCheckBox("automatically fit spline")
        self.auto_fit_checkbox.clicked.connect(self._on_auto_fit)
        self.auto_fit_checkbox.setChecked(self.model.auto_fit_spline)
        self.fit_spline_button = QPushButton("fit spline")
        self.fit_spline_button.pressed.connect(self._on_spline_fit)

        # add the fitting UI to the group_box
        self.fitting_group_box = QGroupBox("fit spline")
        group_box_layout = QVBoxLayout()
        group_box_layout.addWidget(self.auto_fit_checkbox)
        group_box_layout.addWidget(self.fit_spline_button)
        self.fitting_group_box.setLayout(group_box_layout)
        self.fitting_group_box.setVisible(False)

        # add the widget to the layout
        self.layout().addWidget(self.fitting_group_box)

        # connect events to sync changes in the model to the UI
        # self.model.events.active_spline_id.connect(self._update_active_spline_id)

    def _on_spline_fit(self):
        self.model._draw_paths()

    def _on_auto_fit(self):
        """Callback function when the autofit """
        self.model.auto_fit_spline = self.auto_fit_checkbox.isChecked()

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



