import numpy as np

import napari
from napari_threedee.manipulators import PointManipulator

points_data = np.array(
    [
        [0, 0, 0],
        [0, 200, 0],
        [0, 0, 200]
    ]
)

viewer = napari.Viewer(ndisplay=3)
points_layer = viewer.add_points(points_data, size=5)

shapes_layer = viewer.add_shapes(points_data, shape_type='path',  edge_color='magenta')
viewer.layers.selection = [points_layer]


def on_data_update(event):
    shapes_layer.selected_data = {0}
    shapes_layer.remove_selected()

    new_data = points_layer.data
    shapes_layer.add(points_layer.data, shape_type='path', edge_color='magenta')

points_layer.events.data.connect(on_data_update)

point_manipulator = PointManipulator(viewer, points_layer, translator_length=20)

napari.run()