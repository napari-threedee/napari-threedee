import napari
import numpy as np

data = np.random.random((100, 100, 100))

viewer = napari.Viewer(ndisplay=3)
image_layer = viewer.add_image(data, blending='translucent_no_depth')
_ = viewer.add_points(ndim=3)

image_layer.experimental_slicing_plane.enabled = True
image_layer.experimental_slicing_plane.position = np.array(image_layer.data.shape) / 2
napari.run()
