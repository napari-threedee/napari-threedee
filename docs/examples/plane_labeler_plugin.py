"""
Drawing labels on a plane (plugin)
==============================

An example drawing labels on a rendering plane using
the label annotator plugin.
"""
import napari
import numpy as np
from skimage import data


viewer = napari.Viewer(ndisplay=3)

# add a volume
blobs = data.binary_blobs(
    length=64, volume_fraction=0.1, n_dim=3
).astype(np.float32)

# add the image layer in plane depiction mode
plane_layer = viewer.add_image(
    blobs,
    rendering='mip',
    colormap="cyan",
    name='plane',
    depiction='plane',
    opacity=0.5,
    plane={'position': (32, 32, 32), 'normal': (1, 1, 1), 'thickness': 10}
)

# add a labels layer
labels_layer = viewer.add_labels(
    np.zeros_like(blobs).astype(int)
)
labels_layer.n_edit_dimensions = 3

viewer.window.add_plugin_dock_widget(
    "napari-threedee", widget_name="label annotator"
)

napari.run()
