import napari

from napari_threedee._backend.threedee_widget_base import QtThreeDeeWidgetBase

from napari_threedee.annotators.points import PointAnnotator


class QtPointAnnotatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(model_class=PointAnnotator, viewer=viewer)
