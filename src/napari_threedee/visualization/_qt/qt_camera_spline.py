import napari
from napari.utils.events import Event
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QGroupBox, QRadioButton
from qtpy.QtCore import Qt
from superqt.sliders import QLabeledDoubleSlider

from ..._backend.threedee_widget_model import QtThreeDeeWidgetBase
from napari_threedee.visualization.camera_spline import CameraSpline, CameraSplineMode


class QtCameraDirectionControls(QWidget):
    """Widget for setting the camera direction along the spline"""
    def __init__(self, viewer, model, parent=None):
        super().__init__(parent=parent)

        self.viewer = viewer
        self.model = model

        self.set_view_button = QPushButton("set from current view")
        self.set_view_button.clicked.connect(self._on_set_view)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(QLabel("view direction:"))
        self.layout().addWidget(self.set_view_button)

    def _on_set_view(self, event=None):
        """Callback that sets the view direction from the current view"""
        self.model.calculate_transform_from_spline_tangent_to_view_direction()


class QCameraSplineNavationWidget(QWidget):
    """Widget for controlling the camera position along the spline."""
    def __init__(self, viewer, model, parent=None):
        super().__init__(parent=parent)

        self.viewer = viewer
        self.model = model

        # create the slider
        self.spline_slider = QLabeledDoubleSlider()
        self.spline_slider.setMinimum(0)
        self.spline_slider.setMaximum(1)
        self.spline_slider.setOrientation(Qt.Orientation.Horizontal)
        self.spline_slider.valueChanged.connect(self._on_slider_moved)

        # create the view direction controls
        self.camera_direction_controls = QtCameraDirectionControls(viewer=viewer, model=model)

        # create the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.spline_slider)
        self.layout().addWidget(self.camera_direction_controls)

        # initialize interactivity
        self._on_spline_update()

        # connect events
        self.model.events.spline_valid.connect(self._on_spline_update)

    def _on_slider_moved(self):
        tick_position = self.spline_slider.value()
        self.model.current_spline_coordinate = tick_position

    def _on_spline_update(self, event=None):
        """Update the visibility of the spline slider depending if there is a valid spline."""
        if self.model.spline_valid is True:
            self.setDisabled(False)
        else:
            self.setDisabled(True)


class QtCameraSplineControls(QWidget):
    """Widget for annotating and exploring a spline path through a scene."""
    PAN_ZOOM_BUTTON_NAME: str = "pan_zoom"
    ANNOTATE_BUTTON_NAME: str = "annotate"
    EXPLORE_BUTTON_NAME: str = "explore"

    def __init__(self, viewer, model, parent=None):
        super().__init__(parent=parent)

        self.viewer = viewer
        self.model = model

        self.pan_zoom_button = QRadioButton(self.PAN_ZOOM_BUTTON_NAME)
        self.annotate_button = QRadioButton(self.ANNOTATE_BUTTON_NAME)
        self.explore_button = QRadioButton(self.EXPLORE_BUTTON_NAME)


        # setup the layout for the groupbox
        mode_groupbox_layout = QVBoxLayout()
        mode_groupbox_layout.addWidget(self.pan_zoom_button)
        mode_groupbox_layout.addWidget(self.annotate_button)
        mode_groupbox_layout.addWidget(self.explore_button)

        # make the gropubox for the mode
        mode_group_box = QGroupBox("select a mode")
        mode_group_box.setLayout(mode_groupbox_layout)

        self.spline_navigation_widget = QCameraSplineNavationWidget(viewer=viewer, model=model)
        self.spline_navigation_widget.setVisible(False)

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(mode_group_box)
        self.layout().addWidget(self.spline_navigation_widget)

        # connect events
        self.model.events.mode.connect(self._on_mode_changed)
        self.pan_zoom_button.toggled.connect(self._on_mode_button_clicked)
        self.annotate_button.toggled.connect(self._on_mode_button_clicked)
        self.explore_button.toggled.connect(self._on_mode_button_clicked)

        # initialize the groupbox
        self._on_mode_changed()

    def _on_mode_button_clicked(self):
        """Update the mode based on the selection."""
        button = self.sender()
        button_name = button.text()

        if button.isChecked():
            if button_name == self.PAN_ZOOM_BUTTON_NAME:
                self.model.mode = "pan_zoom"
            elif button_name == self.EXPLORE_BUTTON_NAME:
                self.model.mode = "explore"
            elif button_name == self.ANNOTATE_BUTTON_NAME:
                self.model.mode = "annotate"
            else:
                raise ValueError("invalid mode")

    def _on_mode_changed(self, event=None):
        """Update the UI based when the model mode changes."""
        if self.model.mode == CameraSplineMode.PAN_ZOOM:
            self.spline_navigation_widget.setVisible(False)

            # check the button
            self.pan_zoom_button.blockSignals(True)
            self.pan_zoom_button.setChecked(True)
            self.pan_zoom_button.blockSignals(False)
        elif self.model.mode == CameraSplineMode.ANNOTATE:
            self.spline_navigation_widget.setVisible(False)

            # check the button
            self.annotate_button.blockSignals(True)
            self.annotate_button.setChecked(True)
            self.annotate_button.blockSignals(False)
        elif self.model.mode == CameraSplineMode.EXPLORE:
            self.spline_navigation_widget.setVisible(True)
            self.explore_button.blockSignals(True)
            self.explore_button.setChecked(True)
            self.explore_button.blockSignals(False)
        else:
            raise ValueError("invalid mode")


class QtCameraSpline(QtThreeDeeWidgetBase):
    """Container widget for the camera spline controls"""
    DISABLE_STRING: str = "deactivate"
    ENABLE_STRING: str = "activate"

    def __init__(self, viewer: napari.Viewer):
        super().__init__(model_class=CameraSpline, viewer=viewer)

        # create and add the spline widget
        self.spline_widget = QtCameraSplineControls(viewer=viewer, model=self.model, parent=self)
        self.spline_widget.setVisible(False)
        self.layout().addWidget(self.spline_widget)

    def on_activate_button_click(self, event: Event):
        """Callback function when the activate button is clicked"""
        if self.activate_button.isChecked() is True:
            self.model.enabled = True
            self.activate_button.setText(self.DISABLE_STRING)
            self.spline_widget.setVisible(True)
        else:
            self.model.enabled = False
            self.activate_button.setText(self.ENABLE_STRING)
            self.spline_widget.setVisible(False)
