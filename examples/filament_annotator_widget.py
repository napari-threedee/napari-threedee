import napari
import skimage

from napari_threedee.annotators.filament_annotator import FilamentAnnotator, FilamentAnnotatorWidget

viewer = napari.Viewer(ndisplay=3)
blobs = skimage.data.binary_blobs(
    length=64,
    volume_fraction=0.1,
    n_dim=3
).astype(float)

plane_parameters_z = {
    'position': (32, 32, 32),
    'normal': (1, 0, 0),
    'thickness': 10,
    # 'enabled': True
}

plane_parameters_y = {
    'position': (32, 32, 32),
    'normal': (0, 1, 0),
    'thickness': 10,
    # 'enabled': True
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

# viewer.add_image(
#     blobs,
#     name='blue plane',
#     rendering='average',
#     colormap='bop blue',
#     blending='additive',
#     opacity=0.5,
#     experimental_slicing_plane=plane_parameters_y)

viewer.add_points([], ndim=3, face_color='cornflowerblue')

annotator_widget = FilamentAnnotatorWidget(viewer=viewer)
viewer.window.add_dock_widget(widget=annotator_widget)

viewer.camera.angles = (60, 60, 60)
napari.run()
