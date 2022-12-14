import shutil

import zarr
from napari.layers import Layer

from napari_threedee.annotators.io import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY
from napari_threedee.annotators.io._napari import read_n3d_zarr, \
    write_n3d_zarr, IMPLEMENTATIONS


def test_read_and_write_n3d_zarr(valid_n3d_layers, tmp_path):
    # need n3d extension for reading with napari reader
    tmp_path = tmp_path.with_suffix('.n3d')

    for layer in valid_n3d_layers:
        # validate layer
        annotation_type = layer.metadata[N3D_METADATA_KEY][ANNOTATION_TYPE_KEY]
        layer_validator = IMPLEMENTATIONS[annotation_type]['layer_validator']
        layer_validator(layer)

        # write layer
        data, attributes, layer_type = layer.as_layer_data_tuple()
        write_n3d_zarr(
            path=tmp_path,
            data=data,
            attributes=attributes
        )

        # read and validate output as layer and zarr
        assert tmp_path.exists() and tmp_path.is_dir()
        reader = read_n3d_zarr(tmp_path)
        [layer_data_tuple] = reader(tmp_path)
        layer = Layer.create(*layer_data_tuple)
        layer_validator(layer)

        n3d_zarr = zarr.open(tmp_path)
        zarr_validator = IMPLEMENTATIONS[annotation_type]['zarr_validator']
        zarr_validator(n3d_zarr)

        # cleanup for next layer
        shutil.rmtree(tmp_path)

