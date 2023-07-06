import napari

from napari_threedee._backend.threedee_widget_base import QtThreeDeeWidgetBase
from napari_threedee.annotators.label.annotator import PlaneLabeler


class QtLabelAnnotatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(model_class=PlaneLabeler, viewer=viewer)
