import napari
import numpy as np
import skimage

from napari_threedee.annotators import PathAnnotator
from napari_threedee.data_models import N3dPaths, N3dPath

USE_EXISTING_POINTS_LAYER = True

viewer = napari.Viewer(ndisplay=3)
blobs = skimage.data.binary_blobs(
    length=64,
    volume_fraction=0.1,
    n_dim=4
).astype(float)

plane_parameters = {
    'position': (32, 32, 32),
    'normal': (1, 0, 0),
    'thickness': 10,
}

image_layer = viewer.add_image(
    blobs,
    name='orange plane',
    rendering='average',
    colormap='bop orange',
    blending='translucent',
    opacity=0.5,
    depiction='plane',
    plane=plane_parameters
)

if USE_EXISTING_POINTS_LAYER:
    paths = N3dPaths(data=[
        N3dPath(data=np.random.uniform(low=4, high=28, size=(10, 3)))
    ])
    points_layer = paths.as_layer()
else:
    points_layer = None

annotator = PathAnnotator(
    viewer=viewer,
    image_layer=image_layer,
    points_layer=points_layer,
    enabled=True,
)

viewer.camera.angles = (60, 60, 60)
napari.run()
