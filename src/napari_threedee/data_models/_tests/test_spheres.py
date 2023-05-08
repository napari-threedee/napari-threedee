import napari
import napari_threedee as n3d


def test_empty_spheres_as_layer():
    n3d_spheres = n3d.data_models.N3dSpheres(centers=[], radii=[])
    assert n3d_spheres.centers.shape == (0, 3)
    assert n3d_spheres.radii.shape == (0,)
    layer = n3d_spheres.as_layer()
    assert isinstance(layer, napari.layers.Points)
    assert layer.data.shape == (0, 3)
    assert layer.ndim == 3
    assert 'sphere_id' in layer.feature_defaults
    assert 'radius' in layer.feature_defaults
