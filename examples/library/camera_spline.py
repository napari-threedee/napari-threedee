import napari
import skimage

from napari_threedee.visualization._qt.qt_camera_spline import QtCameraSpline

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

viewer.add_image(
    blobs,
    name='orange plane',
    rendering='average',
    colormap='bop orange',
    depiction='plane',
    plane=plane_parameters_z
)

camera_spline = QtCameraSpline(viewer=viewer)
viewer.window.add_dock_widget(camera_spline)

napari.run()
