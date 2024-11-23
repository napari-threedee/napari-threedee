"""
Clip plane manipulator (library)
==========================================

An example controlling the clipping plane manipulator,
using napari-threedee as a library.
"""

import napari

from napari_threedee.manipulators import ClippingPlaneManipulator

viewer = napari.Viewer(ndisplay=3)

membrane, cell_nuclei = viewer.open_sample('napari', 'cells3d')

manipulator = ClippingPlaneManipulator(viewer=viewer, layer=membrane)

membrane.experimental_clipping_planes[0].position = (32, 124, 124)
membrane.experimental_clipping_planes[0].normal = (0, 1, 1)
viewer.layers.selection = {membrane}
viewer.camera.angles = (10, -30, 130)

napari.run()
