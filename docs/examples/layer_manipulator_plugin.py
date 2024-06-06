"""
Layer manipulator (plugin)
==================================

An example controlling the layer manipulator,
using napari-threedee as a napari plugin.
"""
import napari

try:
    from skimage.data import binary_blobs
except ImportError:
    raise ImportError("This example requires scikit-image. pip install scikit-image")


image = binary_blobs(
    50,
    n_dim=3,
    blob_size_fraction=0.3,
    volume_fraction=0.1,
)

viewer = napari.view_image(
    image,
    colormap="magenta",
    opacity=0.8,
    blending="additive",
    rendering="iso",
    iso_threshold=0,
    name="image_0",
)
image_layer_0 = viewer.layers[0]

# add an addition layer that is shifted
image_layer_1 = viewer.add_image(
    image.copy(),
    colormap="green",
    opacity=0.8,
    blending="additive",
    rendering="iso",
    iso_threshold=0,
    name="image_1",
    translate=(20, 20, 20),
)

viewer.dims.ndisplay = 3

viewer.window.add_plugin_dock_widget(
    plugin_name="napari-threedee", widget_name="layer manipulator"
)

napari.run()
