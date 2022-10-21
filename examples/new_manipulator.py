import napari
import numpy as np

from napari_threedee._infrastructure.manipulator.napari_manipulator import NapariManipulator

viewer = napari.Viewer(ndisplay=3)
viewer.axes.visible = True
image_layer = viewer.add_image(np.zeros((50, 50, 50)))

manipulator = NapariManipulator(
    rotator_axes='xyz',
    translator_axes='xyz',
    viewer=viewer,
    layer=image_layer
)

viewer.camera.center = (24.5, 24.5, 24.5)
viewer.camera.zoom = 9.5
viewer.camera.angles = (10.496500306786134, 35.473826801211615, -158.2263269182936)

napari.run()
