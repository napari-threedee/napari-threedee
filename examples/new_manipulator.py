import napari
import numpy as np

from napari_threedee.manipulators.new_manipulator import Manipulator

viewer = napari.Viewer()

# lines = 50 * np.array(
#     [
#         [
#             [0, 0, 0],
#             [1, 0, 0]
#         ],
#         [
#             [0, 0, 0],
#             [0, 1, 0]
#         ],
#         [
#             [0, 0, 0],
#             [0, 0, 1]
#         ]
#     ]
# )
#
# layer = viewer.add_shapes(lines, shape_type="line")

layer = viewer.add_image(np.zeros((50, 50, 50)))

viewer.dims.ndisplay = 3

manipulator = Manipulator(
    rotator_axis_indices=np.array([0, 1]),
    translator_axis_indices=np.array([0]),
    viewer=viewer,
    layer=layer
)

viewer.camera.center = (24.5, 24.5, 24.5)
viewer.camera.zoom = 9.5
viewer.camera.angles = (10.496500306786134, 35.473826801211615, -158.2263269182936)

napari.run()
