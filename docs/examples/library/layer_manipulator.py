"""
Layer manipulator example (library)
===================================

An example controlling the layer manipulator,
using napari-threedee as a library.
"""
import napari
import numpy as np

from napari_threedee.manipulators._qt import QtLayerManipulatorWidget

data = np.random.random((100, 100, 100))

viewer = napari.view_image(data)
image_layer = viewer.layers[0]

viewer.dims.ndisplay = 3

widget = QtLayerManipulatorWidget(viewer)
viewer.window.add_dock_widget(widget)

napari.run()
