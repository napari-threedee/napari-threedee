import numpy as np

from vispy.scene import Compound, Mesh, Line


class FinitePlaneVisual(Compound):
    def __init__(self, parent):
        super().__init__([Mesh(), Line()], parent=parent)
        plane_corners = np.array(
            [[0.5, 0.5, 0],
             [0.5, -0.5, 0],
             [-0.5, -0.5, 0],
             [-0.5, 0.5, 0]]
        )
        plane_corner_connections = np.array(
            [[0, 1],
             [1, 2],
             [2, 3],
             [3, 0]]
        )

        # set the data on the line visual
        self.line_visual.set_data(
            pos=plane_corners,
            color='blue',
            width=3,
            connect=plane_corner_connections,
        )

    @property
    def mesh_visual(self) -> Mesh:
        return self._subvisuals[0]

    @property
    def line_visual(self) -> Line:
        return self._subvisuals[1]


if __name__ == '__main__':
    import napari
    from napari_threedee.utils.napari_utils import get_vispy_node

    image = np.zeros((10, 10, 10))
    viewer = napari.Viewer(ndisplay=3)
    image_layer = viewer.add_image(image)
    vispy_node = get_vispy_node(viewer, layer=image_layer)

    plane_visual = FinitePlaneVisual(parent=vispy_node)
    napari.run()
