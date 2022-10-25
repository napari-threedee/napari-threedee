from napari_threedee.manipulators._qt import QtRenderPlaneManipulatorWidget

import napari
from skimage import data

viewer = napari.Viewer(ndisplay=3)
blobs = data.binary_blobs(
    length=64, volume_fraction=0.1, n_dim=3
).astype(float)

plane_parameters = {
    'position': (32, 32, 32),
    'normal': (1, 0, 0),
    'thickness': 10,
}

plane_layer = viewer.add_image(
    blobs,
    rendering='average',
    name='plane',
    colormap='bop orange',
    blending='additive',
    opacity=0.5,
    depiction='plane',
    plane=plane_parameters
)
volume_layer = viewer.add_image(
    blobs, rendering='mip', name='volume', blending='additive', opacity=0.25
)

widget = QtRenderPlaneManipulatorWidget(viewer)
viewer.window.add_dock_widget(widget)
napari.run()
