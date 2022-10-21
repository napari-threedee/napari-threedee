import napari

from napari_threedee._backend.threedee_widget_model import QtThreeDeeWidgetBase

from napari_threedee.annotators.plane_point_annotator import PlanePointAnnotator


class QtPlanePointAnnotatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(model_class=PlanePointAnnotator, viewer=viewer)
