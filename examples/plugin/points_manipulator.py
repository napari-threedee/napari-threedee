"""Example of using the points manipulator to re-position points.

Enter the point selection mode and click on a point to activate the manipulator.
Click and drag the arms of the manipulator to move the point.
When a point is selected, hold space to rotate the view without losing the selection.
"""

import numpy as np

import napari

points_data = np.array(
    [
        [0, 0, 0],
        [0, 200, 0],
        [0, 0, 200]
    ]
)

viewer = napari.Viewer(ndisplay=3)
points_layer = viewer.add_points(points_data, size=5)

napari.run()
