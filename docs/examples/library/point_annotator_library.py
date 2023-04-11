"""
Point annotator (library)
=================================

An example controlling the point annotator,
using napari-threedee as a library.
"""
import napari
from skimage import data

from napari_threedee.annotators import PointAnnotator


# create napari viewer
viewer = napari.Viewer(ndisplay=3)

# generate 3d image data
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

# add points layer to viewer
points_layer = viewer.add_points(
    data=[],
    size=5,
    face_color='cornflowerblue',
    ndim=3
)

# create the point annotator
annotator = PointAnnotator(
    viewer=viewer,
    image_layer=image_layer,
    points_layer=points_layer,
    enabled=True,
)

# run napari
viewer.layers.selection = [image_layer]
viewer.axes.visible = True
viewer.camera.angles = (-15, 25, -30)
viewer.camera.zoom *= 0.5
napari.run()
