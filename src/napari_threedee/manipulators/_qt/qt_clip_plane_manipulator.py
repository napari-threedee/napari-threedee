import napari

from ..._backend.threedee_widget_base import QtThreeDeeWidgetBase
from ..clip_plane_manipulator import ClippingPlaneManipulator


class QtClippingPlaneManipulatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer, *args, **kwargs):
        super().__init__(model_class=ClippingPlaneManipulator, viewer=viewer, *args, **kwargs)