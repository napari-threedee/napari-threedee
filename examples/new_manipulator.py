import napari
import numpy as np

from napari_threedee._manipulator_base import Manipulator

viewer = napari.Viewer()

layer = viewer.add_image(np.zeros((50, 50, 50)))

viewer.dims.ndisplay = 3

manipulator = Manipulator(
    rotator_axes='xyz',
    translator_axes='xyz',
    viewer=viewer,
    layer=layer
)

viewer.camera.center = (24.5, 24.5, 24.5)
viewer.camera.zoom = 9.5
viewer.camera.angles = (10.496500306786134, 35.473826801211615, -158.2263269182936)

napari.run()
