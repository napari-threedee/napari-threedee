"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import napari
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory
from functools import partial

from .mouse_callbacks import add_point_on_plane, shift_plane_along_normal

class QtPlaneControls(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.mouse_drag_callbacks.append(
            partial(
                add_point_on_plane,
                points_layer=viewer.layers[1],
                plane_layer=viewer.layers[0],
            )
        )
        self.viewer.mouse_drag_callbacks.append(
            partial(
                shift_plane_along_normal,
                layer=viewer.layers[0]
            )
        )

        btn = QPushButton("Useless button!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("lols")


@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return QtPlaneControls,
