"""This implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html
"""
from napari_plugin_engine import napari_hook_implementation

from .manipulators.qt_manipulators import QtRenderPlaneManipulatorWidget, QtPointManipulatorWidget, QtLayerManipulatorWidget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [
        (QtRenderPlaneManipulatorWidget, {"name": "Render plane manipulator"}),
        (QtPointManipulatorWidget, {"name": "Point manipulator"}),
        (QtLayerManipulatorWidget, {"name": "Layer manipulator"}),
    ]
