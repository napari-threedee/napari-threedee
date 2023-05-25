try:
    import igl
    IGL_INSTALLED = True
except ModuleNotFoundError:
    IGL_INSTALLED = False

from typing import List

import napari
from napari.qt.threading import FunctionWorker, thread_worker
from napari.utils import progress
from magicgui import magicgui, widgets, magic_factory
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QLabel

from napari_threedee.visualization.lighting_control import LightingControl


class QtLightingControlWidget(QWidget):
    _ENABLE_TEXT = 'start following camera'
    _DISABLE_TEXT = 'stop following camera'
    _TOOLTIP = 'viewer must be in 3D mode'

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer
        self.model = LightingControl(viewer=viewer)

        # create layer selection widget
        self._layer_selection_widget = magicgui(
            self.model.set_layers,
            layers={
                'widget_type': 'Select',
                'choices': self._get_layers
            },
            auto_call=True
        )

        self._viewer.dims.events.ndisplay.connect(self._on_ndisplay_change)
        self._viewer.layers.events.inserted.connect(
            self._layer_selection_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._layer_selection_widget.reset_choices
        )

        # create set lighing widget
        self._lighting_button = QPushButton(self._ENABLE_TEXT, self)
        self._lighting_button.setChecked(False)
        self._lighting_button.clicked.connect(self._on_lighting_clicked)
        self._update_lighting_button()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._layer_selection_widget.native)
        self.layout().addWidget(self._lighting_button)

    def _update_lighting_button(self):
        viewer_is_3d = self._viewer.dims.ndisplay == 3
        self._lighting_button.setCheckable(viewer_is_3d)
        self._lighting_button.setEnabled(viewer_is_3d)
        self._lighting_button.setToolTip("" if viewer_is_3d else self._TOOLTIP)

    def _on_lighting_clicked(self, event):
        if self._lighting_button.isChecked() is True:
            self.model.enabled = True
            self._lighting_button.setText(self._DISABLE_TEXT)
        else:
            self.model.enabled = False
            self._lighting_button.setText(self._ENABLE_TEXT)

    def _on_ndisplay_change(self, event):
        ndisplay = event.value
        if ndisplay == 2 and self._lighting_button.isChecked() is True:
            self._lighting_button.click()
        self._update_lighting_button()

    def _get_layers(self, widget):
        return [layer for layer in self._viewer.layers if isinstance(layer, napari.layers.Surface)]


class QtAmbientOcclusionWidget(QWidget):

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer

        # initialize state variables
        self.selected_layers = []

        # layers currently with ambient occlusion activated
        self.current_ao_layers = []

        # store the original values so they can be reset
        self.original_data = {}

        # create layer selection widget
        self._layer_selection_widget = magicgui(
            self.set_layers,
            layers={
                'widget_type': 'Select',
                'choices': self._get_layers
            },
            auto_call=True
        )

        self._viewer.layers.events.inserted.connect(
            self._layer_selection_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._layer_selection_widget.reset_choices
        )

        if IGL_INSTALLED:
            self._ambient_occlusion_widget = QPushButton("update ambient occlusion")
            self._ambient_occlusion_widget.pressed.connect(self._set_ambient_occlusion)
        else:
            self._ambient_occlusion_widget = QLabel("IGL is not installed. pip install libigl")

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._layer_selection_widget.native)
        self.layout().addWidget(self._ambient_occlusion_widget)

    def _get_layers(self, widget):
        return [layer for layer in self._viewer.layers if isinstance(layer, napari.layers.Surface)]

    def set_layers(self, layers: List[napari.layers.Surface]):
        self.selected_layers = layers

    def _set_ambient_occlusion(self) -> None:
        current_selection = set(self.selected_layers)
        current_ao = set(self.current_ao_layers)

        # determine which layers need to be updated
        layers_to_add_ao_set = current_selection.difference(current_ao)
        layers_to_add_ao = list(layers_to_add_ao_set)
        layers_to_remove_ao = list(current_ao.difference(current_selection))
        self.current_ao_layers = list(layers_to_add_ao_set.union(current_ao.intersection(current_selection)))

        with progress(layers_to_add_ao) as adding_progress:
            for layer in adding_progress:
                vertices, faces, values = layer.data
                faces = faces.astype(int)
                vertex_normals = igl.per_vertex_normals(vertices, faces)
                ao = igl.ambient_occlusion(vertices, faces, vertices, vertex_normals, 20)
                attenuation_factors = 1 - ao
                attenuated_values = attenuation_factors * values

                layer.data = (vertices, faces, attenuated_values)

                self.original_data.update({layer: (vertices, faces, values)})

        with progress(layers_to_remove_ao) as removing_progress:
            for layer in removing_progress:
                # remove the ao
                layer_data = self.original_data.pop(layer)
                layer.data = layer_data
