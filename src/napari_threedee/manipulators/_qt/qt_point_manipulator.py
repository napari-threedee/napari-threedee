import napari

from ..point_manipulator import PointManipulator
from ..._backend.threedee_widget_model import QtThreeDeeWidgetBase


class QtPointManipulatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer, *args, **kwargs):
        super().__init__(model_class=PointManipulator, viewer=viewer, *args, **kwargs)
