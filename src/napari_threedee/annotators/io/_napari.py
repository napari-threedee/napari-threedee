import os
from typing import List, Callable, Optional

import numpy as np
import zarr
from napari.layers import Layer
from napari.types import LayerDataTuple

from . import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY
from ..spline_annotator import SplineAnnotator
from ..spheres import SphereAnnotator
from ..points import PointAnnotator
import napari_threedee.annotators.io.sphere as sphere_io
import napari_threedee.annotators.io.spline as spline_io
import napari_threedee.annotators.io.point as point_io

IMPLEMENTATIONS: dict = {
    SphereAnnotator.ANNOTATION_TYPE: {
        'reader': sphere_io.n3d_zarr_to_layer_data_tuple,
        'writer': sphere_io.layer_to_n3d_zarr,
        'layer_validator': sphere_io.validate_layer,
        'zarr_validator': sphere_io.validate_zarr,
    },
    SplineAnnotator.ANNOTATION_TYPE: {
        'reader': spline_io.n3d_zarr_to_layer_data_tuple,
        'writer': spline_io.layer_to_n3d_zarr,
        'layer_validator': spline_io.validate_layer,
        'zarr_validator': spline_io.validate_zarr,
    },
    PointAnnotator.ANNOTATION_TYPE: {
        'reader': point_io.n3d_zarr_to_layer_data_tuple,
        'writer': point_io.layer_to_n3d_zarr,
        'layer_validator': point_io.validate_layer,
        'zarr_validator': point_io.validate_n3d_zarr,
    }
}


def write_n3d_zarr(
    path: str, data: np.ndarray, attributes: dict
) -> List[str]:
    """Write an n3d zarr file from napari layer data."""
    layer = Layer.create(data=data, meta=attributes, layer_type='points')
    annotation_type = layer.metadata[N3D_METADATA_KEY][ANNOTATION_TYPE_KEY]
    to_zarr = IMPLEMENTATIONS[annotation_type]['writer']
    to_zarr(layer, path)
    return [path]


def read_n3d_zarr(path: os.PathLike) -> Optional[Callable]:
    """Return a function for reading .n3d files if possible."""
    return _read_n3d_zarr if str(path).endswith('.n3d') else None


def _read_n3d_zarr(path: os.PathLike) -> List[LayerDataTuple]:
    """Dispatches to specific readers based on annotation type."""
    n3d_zarr = zarr.open_array(path)
    annotation_type = n3d_zarr.attrs[ANNOTATION_TYPE_KEY]
    to_layer_data_tuple = IMPLEMENTATIONS[annotation_type]['reader']
    return [to_layer_data_tuple(n3d_zarr)]
