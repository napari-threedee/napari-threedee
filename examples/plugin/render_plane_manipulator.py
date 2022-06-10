import napari
from skimage import data


viewer = napari.Viewer(ndisplay=3)

# add a volume
blobs = data.binary_blobs(
    length=64, volume_fraction=0.1, n_dim=3
).astype(float)
volume_layer = viewer.add_image(
    blobs, rendering='mip', name='volume', blending='additive', opacity=0.25
)

# add the same volume and render as plane
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

viewer.window.add_plugin_dock_widget(
    plugin_name="napari-threedee", widget_name="render plane manipulator"
)

napari.run()
