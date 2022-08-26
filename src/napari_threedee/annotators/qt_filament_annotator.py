import napari

from ..base import QtThreeDeeWidgetBase

from .filament_annotator import FilamentAnnotator


class QtFilamentAnnotatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(model=FilamentAnnotator, viewer=viewer)
