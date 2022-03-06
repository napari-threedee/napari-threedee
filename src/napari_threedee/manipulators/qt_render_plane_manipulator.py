import napari

from .render_plane_manipulator import RenderPlaneManipulator
from ..base import QtThreeDeeWidgetBase


class QtRenderPlaneManipulatorWidget(QtThreeDeeWidgetBase):
    def __init__(self,viewer: napari.Viewer, *args, **kwargs):
        super().__init__(model=RenderPlaneManipulator, viewer=viewer, *args, **kwargs)