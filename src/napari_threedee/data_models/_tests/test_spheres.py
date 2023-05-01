import napari_threedee as n3d


def test_empty_spheres():
    spheres = n3d.data_models.N3dSpheres(centers=[], radii=[])
    assert spheres.centers.shape == (0, 3)
    assert spheres.radii.shape == (0, )