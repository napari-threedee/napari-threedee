import napari
import skimage


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

viewer.add_points([], ndim=4, face_color='cornflowerblue')

viewer.window.add_plugin_dock_widget(
    plugin_name="napari-threedee", widget_name="plane point annotator"
)

viewer.camera.angles = (60, 60, 60)
napari.run()
