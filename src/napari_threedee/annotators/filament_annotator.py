from typing import Optional

from ..base import ThreeDeeModel, QtThreeDeeWidgetBase
import napari


class FilamentAnnotator(ThreeDeeModel):
    def __init__(
            self,
            viewer: napari.Viewer,
            image_layer: Optional[napari.layers.Image] = None
    ):
        self.viewer = viewer
        self.image_layer = image_layer
        self.points_layer = viewer.add_points(
            [], ndim=3, face_color='cornflowerblue', features={''}
        )
        self.active_filament_idx: int = 0

    def set_layers(self, image_layer: napari.layers.Image):
        self.image_layer = image_layer

    def _on_enable(self):
        pass

    def _on_disable(self):
        pass

    def _start_new_filament(self):
        self.active_filament_idx += 1


class FilamentAnnotatorWidget(QtThreeDeeWidgetBase):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(model=FilamentAnnotator, viewer=viewer)

