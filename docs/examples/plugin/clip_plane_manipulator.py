"""
Clipping plane manipulator (plugin)
==========================================

An example controlling the clipping plane manipulator,
using napari-threedee as a napari plugin.
"""

import napari

viewer = napari.Viewer(ndisplay=3)

# Add some 3D data
membrane, cell_nuclei = viewer.open_sample('napari', 'cells3d')

viewer.layers.selection = {membrane}
viewer.camera.angles = (10, -30, 130)

viewer.window.add_plugin_dock_widget(
    plugin_name="napari-threedee", widget_name="clipping plane manipulator"
)

napari.run()
