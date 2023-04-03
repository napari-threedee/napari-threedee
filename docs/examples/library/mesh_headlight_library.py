"""
Mesh headlight example (library)
================================

An example controlling the mesh headlight,
using napari-threedee as a library.
"""
import napari
import numpy as np
from vispy.io import load_data_file, read_mesh

from napari_threedee.visualization._qt.qt_lighting_control import QtLightingControlWidget


# Fetch datasets - this will download dataset if datasets are not found
vertices, faces, _, _ = read_mesh(load_data_file('orig/triceratops.obj.gz'))

# put the mesh right side up, scale it up (napari#3477) and fix faces handedness
vertices *= -100
faces = faces[:, ::-1]
vertex_values = np.ones((len(vertices),))

# create an empty viewer
viewer = napari.Viewer(ndisplay=3)

# add the mesh
viewer.add_surface((vertices, faces, vertex_values), name='triceratops')

lighting_control_widget = QtLightingControlWidget(viewer=viewer)
viewer.window.add_dock_widget(widget=lighting_control_widget)

napari.run()
