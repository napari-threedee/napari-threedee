import napari
import skimage

from napari_threedee.annotators._qt import QtSplineAnnotatorWidget


viewer = napari.Viewer(ndisplay=3)
blobs = skimage.data.binary_blobs(
    length=64,
    volume_fraction=0.1,
    n_dim=4
).astype(float)

plane_parameters_z = {
    'position': (32, 32, 32),
    'normal': (1, 0, 0),
    'thickness': 10,
}

plane_parameters_y = {
    'position': (32, 32, 32),
    'normal': (0, 1, 0),
    'thickness': 10,
}

viewer.add_image(
    blobs,
    name='orange plane',
    rendering='average',
    colormap='bop orange',
    blending='additive',
    opacity=0.5,
    depiction='plane',
    plane=plane_parameters_z)

viewer.add_image(
    blobs,
    name='blue plane',
    rendering='average',
    colormap='bop blue',
    blending='additive',
    opacity=0.5,
    depiction='plane',
    plane=plane_parameters_y)

spline_annotator = QtSplineAnnotatorWidget(viewer=viewer)
viewer.window.add_dock_widget(spline_annotator)


viewer.camera.angles = (60, 60, 60)
napari.run()
