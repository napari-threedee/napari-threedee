import napari
from magicgui import magicgui
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout

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
