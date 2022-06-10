from napari_threedee.annotators import PlanePointAnnotator

import napari
from skimage import data

viewer = napari.Viewer(ndisplay=3)
blobs = data.binary_blobs(
    length=64, volume_fraction=0.1, n_dim=3
).astype(float)

# add a volume and render as plane
# plane should be in 'additive' blending mode or depth looks all wrong
plane_parameters = {
    'position': (32, 32, 32),
    'normal': (1, 0, 0),
    'thickness': 10,
    'enabled': True
}

plane_layer = viewer.add_image(
    blobs,
    rendering='average',
    name='plane',
    colormap='bop orange',
    blending='additive',
    opacity=0.5,
    experimental_slicing_plane=plane_parameters
)

points_layer = viewer.add_points([], size=5, face_color='cornflowerblue', ndim=3)

annotator = PlanePointAnnotator(
    viewer=viewer, image_layer=plane_layer, points_layer=points_layer
)

napari.run()