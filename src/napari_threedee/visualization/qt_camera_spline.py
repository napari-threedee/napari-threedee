import napari
from qtpy.QtWidgets import QPushButton
from qtpy.QtCore import Qt
from superqt.sliders import QLabeledDoubleSlider

from ..base import QtThreeDeeWidgetBase
from .camera_spline import CameraSpline

class QtCameraSpline(QtThreeDeeWidgetBase):

    DISABLE_STRING: str = "disable spline annotation"
    ENABLE_STRING: str = "enable spline annotation"

    def __init__(self, viewer: napari.Viewer):
        super().__init__(model=CameraSpline, viewer=viewer)

        self.annotate_push_button = QPushButton("annotate spline")
        self.annotate_push_button.setCheckable(True)
        self.annotate_push_button.clicked.connect(self._on_annotate_clicked)

        self.spline_slider = QLabeledDoubleSlider()
        self.spline_slider.setMinimum(0)
        self.spline_slider.setMaximum(1)
        self.spline_slider.setOrientation(Qt.Orientation.Horizontal)
        self.spline_slider.valueChanged.connect(self._on_slider_moved)

        self.layout().addWidget(self.annotate_push_button)
        self.layout().addWidget(self.spline_slider)

    def _on_annotate_clicked(self):
        if self.annotate_push_button.isChecked() is True:
            self.annotate_push_button.setText(self.DISABLE_STRING)
            self.model.start_spline_annotation()
        else:
            self.annotate_push_button.setText(self.ENABLE_STRING)
            self.model.stop_spline_annotation()

    def _on_slider_moved(self):
        tick_position = self.spline_slider.value()
        print(tick_position)
        self.model.set_camera_position(tick_position)
