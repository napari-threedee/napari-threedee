import napari

from .._base_model import QtThreeDeeWidgetBase

from napari_threedee.annotators.sphere_annotator import SphereAnnotator


class QtSphereAnnotatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(model=SphereAnnotator, viewer=viewer)
