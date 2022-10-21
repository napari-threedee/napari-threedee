import napari

from ..layer_manipulator import LayerManipulator
from ..._backend.threedee_widget_model import QtThreeDeeWidgetBase


class QtLayerManipulatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer, *args, **kwargs):
        super().__init__(model_class=LayerManipulator, viewer=viewer, *args, **kwargs)
