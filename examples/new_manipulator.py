import napari
import numpy as np

from napari_threedee.manipulators.new_manipulator import Manipulator

viewer = napari.Viewer()

# im = np.random.random((50, 50, 50))
im = np.zeros((50, 50, 50))

image_layer = viewer.add_image(im)

viewer.dims.ndisplay = 3

manipulator = Manipulator(
    viewer=viewer,
    layer=image_layer
)

viewer.camera.center = (24.5, 24.5, 24.5)
viewer.camera.zoom = 9.5
viewer.camera.angles = (10.496500306786134, 35.473826801211615, -158.2263269182936)

napari.run()
