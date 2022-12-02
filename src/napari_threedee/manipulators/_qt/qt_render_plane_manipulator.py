import napari

from ..._backend.threedee_widget_model import QtThreeDeeWidgetBase
from ..render_plane_manipulator import RenderPlaneManipulator


class QtRenderPlaneManipulatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer, *args, **kwargs):
        super().__init__(model_class=RenderPlaneManipulator, viewer=viewer, *args, **kwargs)