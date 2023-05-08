"""
Sphere annotator (library)
==================================

An example controlling the sphere annotator,
using napari-threedee as a library.
"""
import napari
import numpy as np
from skimage import data

from napari_threedee.data_models import N3dSpheres
from napari_threedee.annotators.spheres import SphereAnnotator

CREATE_LAYER_FROM_EXISTING_DATA = True

# create napari viewer
viewer = napari.Viewer(ndisplay=3)

# generate image data
blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(float)

# add image layer to viewer (rendering as a plane)
image_layer = viewer.add_image(
    blobs,
    rendering='average',
    name='plane',
    colormap='bop orange',
    blending='translucent',
    opacity=0.5,
    depiction='plane',
    plane={
        'position': (32, 32, 32),
        'normal': (1, 0, 0),
        'thickness': 10,
    },
)

# optionally create an n3d compatible points layer from existing data
if CREATE_LAYER_FROM_EXISTING_DATA is True:
    centers = np.random.uniform(0, 64, size=(10, 3))
    radii = np.random.uniform(1, 10, size=(10,))
    points_layer = N3dSpheres(centers=centers, radii=radii).as_layer()
else:
    points_layer = None

# create the annotator
annotator = SphereAnnotator(
    viewer=viewer,
    image_layer=image_layer,
    points_layer=points_layer,
    enabled=True
)

# run napari
viewer.axes.visible = True
viewer.axes.labels = False
viewer.camera.angles = (-15, 25, -30)
viewer.camera.zoom *= 0.5
napari.run()
