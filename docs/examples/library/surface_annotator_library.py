"""
Surface annotator (library)
===================================

An example controlling the surface annotator,
using napari-threedee as a library.
"""
import napari
from skimage import data

import napari_threedee as n3d

# create napari viewer
viewer = napari.Viewer(ndisplay=3)

# generate 3d image data
blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(float)

# add image layer to viewer (rendering as plane)
image_layer = viewer.add_image(
    blobs,
    name='orange plane',
    rendering='average',
    colormap='bop orange',
    blending='translucent',
    opacity=0.5,
    depiction='plane',
    plane={
        'position': (32, 32, 32),
        'normal': (1, 0, 0),
        'thickness': 10,
    })

# create annotator
annotator = n3d.annotators.SurfaceAnnotator(
    viewer=viewer, image_layer=image_layer, enabled=True
)

# run napari
viewer.layers.selection = [image_layer]
viewer.axes.visible = True
viewer.axes.labels = False
viewer.camera.angles = (-15, 25, -30)
viewer.camera.zoom *= 0.5
napari.run()
