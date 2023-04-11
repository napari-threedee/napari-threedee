import napari

from ..._backend import QtThreeDeeWidgetBase

from napari_threedee.annotators.spheres import SphereAnnotator


class QtSphereAnnotatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(model_class=SphereAnnotator, viewer=viewer)
