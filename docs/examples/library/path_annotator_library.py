"""
Path point annotator (library)
==============================

An example controlling the path annotator,
using napari-threedee as a library.
"""

import napari
import numpy as np
from skimage import data

from napari_threedee.annotators import PathAnnotator
from napari_threedee.data_models import N3dPaths, N3dPath

CREATE_LAYER_FROM_EXISTING_DATA = True

# create napari viewer
viewer = napari.Viewer(ndisplay=3)

# generate image data
blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=4).astype(float)

# add image layer to viewer (rendering as a plane)
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
    }
)

# optionally create an n3d compatible points layer from existing data
if CREATE_LAYER_FROM_EXISTING_DATA is True:
    path = N3dPath(data=np.random.uniform(low=8, high=56, size=(10, 3)))
    points_layer = N3dPaths(data=[path]).as_layer()  # list of N3dPath
else:
    points_layer = None

# create the path annotator
annotator = PathAnnotator(
    viewer=viewer,
    image_layer=image_layer,
    points_layer=points_layer,
    enabled=True,
)

# run napari
viewer.axes.visible = True
viewer.axes.labels = False
viewer.camera.angles = (-15, 25, -30)
viewer.camera.zoom *= 0.5
napari.run()
