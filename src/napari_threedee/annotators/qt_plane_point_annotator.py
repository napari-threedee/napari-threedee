import napari

from .._base_model import QtThreeDeeWidgetBase

from .plane_point_annotator import PlanePointAnnotator


class QtPlanePointAnnotatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(model=PlanePointAnnotator, viewer=viewer)
