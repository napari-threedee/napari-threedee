import napari_threedee as n3d


def test_empty_points():
    points = n3d.data_models.N3dPoints(data=[])
    assert points.data.shape == (0, 3)
