"""
Sphere annotator example (library)
==================================

An example controlling the sphere annotator,
using napari-threedee as a library.
"""
from napari_threedee.annotators._qt.qt_sphere_annotator import \
    QtSphereAnnotatorWidget

import napari
from skimage import data

viewer = napari.Viewer(ndisplay=3)
blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(float)

# add a volume and render as plane
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
    blending='translucent',
    opacity=0.5,
    depiction='plane',
    plane=plane_parameters,
)

annotator = QtSphereAnnotatorWidget(
    viewer=viewer
)
viewer.window.add_dock_widget(annotator)
napari.run()
