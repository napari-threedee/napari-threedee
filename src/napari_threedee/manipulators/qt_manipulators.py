import napari

from .layer_manipulator import LayerManipulator
from .render_plane_manipulator import RenderPlaneManipulator
from .point_manipulator import PointManipulator
from ..base import QtThreeDeeWidgetBase


class QtRenderPlaneManipulatorWidget(QtThreeDeeWidgetBase):
    def __init__(self,viewer: napari.Viewer, *args, **kwargs):
        super().__init__(model=RenderPlaneManipulator, viewer=viewer, *args, **kwargs)

class QtPointManipulatorWidget(QtThreeDeeWidgetBase):
    def __init__(self,viewer: napari.Viewer, *args, **kwargs):
        super().__init__(model=PointManipulator, viewer=viewer, *args, **kwargs)

class QtLayerManipulatorWidget(QtThreeDeeWidgetBase):
    def __init__(self,viewer: napari.Viewer, *args, **kwargs):
        super().__init__(model=LayerManipulator, viewer=viewer, *args, **kwargs)
