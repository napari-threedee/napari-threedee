import napari
import numpy as np


data = np.random.random((100, 100, 100))

viewer = napari.view_image(data)
image_layer = viewer.layers[0]

viewer.dims.ndisplay = 3

viewer.window.add_plugin_dock_widget(
    plugin_name="napari-threedee", widget_name="layer manipulator"
)

napari.run()
