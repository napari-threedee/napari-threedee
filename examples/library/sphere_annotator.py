import numpy as np

from napari_threedee.annotators.spheres import SphereAnnotator
from napari_threedee.data_models import N3dSpheres

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

centers = np.random.uniform(0, 64, size=(10, 3))
radii = np.random.uniform(1, 10, size=(10, ))
points_layer = N3dSpheres(centers=centers, radii=radii).as_layer()
viewer.add_layer(points_layer)

annotator = SphereAnnotator(
    viewer=viewer,
    image_layer=plane_layer,
    points_layer=points_layer,
    enabled=True
)

napari.run()
