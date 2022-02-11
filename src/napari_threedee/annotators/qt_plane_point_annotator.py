from magicgui import magicgui
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from napari.layers import Image, Points
from typing import List
import napari
from .plane_point_annotator import PlanePointAnnotator



class PlanePointAnnotatorWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer
        self.annotator = PlanePointAnnotator(self._viewer)

        self._layer_selection_widget = magicgui(
            self.update_layers,
            image_layer={'choices': self._get_image_layers},
            points_layer={'choices': self._get_points_layers},
            auto_call=True
        )
        self._annotating_button = QPushButton('start annotating', self)
        self._annotating_button.setCheckable(True)
        self._annotating_button.setChecked(False)
        self._annotating_button.clicked.connect(self._on_annotating_clicked)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._layer_selection_widget.native)
        self.layout().addWidget(self._annotating_button)

        self._layer_selection_widget()  # call with auto-populated layers

    def update_layers(self, image_layer: Image, points_layer: Points):
        self.annotator.points_layer = points_layer
        self.annotator.image_layer = image_layer

    def _on_annotating_clicked(self, event):
        if self._annotating_button.isChecked() is True:
            self.annotator.annotating = True
            self._annotating_button.setText('stop annotating')
        else:
            self.annotator.annotating = False
            self._annotating_button.setText('start annotating')

    def _get_image_layers(self, combo_widget) -> List[Image]:
        return [layer for layer in self._viewer.layers if isinstance(layer, Image)]

    def _get_points_layers(self, combo_widget) -> List[Points]:
        return [layer for layer in self._viewer.layers if isinstance(layer, Points)]
