from typing import List

import napari.layers
import numpy as np
import pytest
import zarr
from napari.layers import Points

import napari_threedee.annotators.spheres.constants
import napari_threedee.annotators.splines.constants
from napari_threedee.annotators import SplineAnnotator, SphereAnnotator, \
    PointAnnotator
from napari_threedee.annotators.io import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY


@pytest.fixture
def valid_n3d_layers() -> List[napari.layers.Layer]:
    return [
        _valid_n3d_spline_layer(),
        _valid_n3d_sphere_layer(),
        _valid_n3d_point_layer(),
    ]


def _valid_n3d_spline_layer() -> napari.layers.Points:
    layer = napari.layers.Points(
        data=np.random.normal(size=(2, 3)),
        features={
            napari_threedee.annotators.splines.constants.SPLINE_ID_FEATURES_KEY: [0, 1]},
        metadata={
            N3D_METADATA_KEY: {
                ANNOTATION_TYPE_KEY: SplineAnnotator.ANNOTATION_TYPE
            }
        }
    )
    return layer


@pytest.fixture
def valid_n3d_spline_layer() -> napari.layers.Points:
    return _valid_n3d_spline_layer()


def _valid_n3d_spline_zarr() -> zarr.Array:
    n3d_zarr = zarr.array(np.random.normal(size=(2, 3)))
    n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = SplineAnnotator.ANNOTATION_TYPE
    n3d_zarr.attrs[napari_threedee.annotators.splines.constants.SPLINE_ID_FEATURES_KEY] = [0, 1]
    return n3d_zarr


@pytest.fixture
def valid_n3d_spline_zarr() -> zarr.Array:
    return _valid_n3d_spline_zarr()


def _valid_n3d_sphere_layer() -> napari.layers.Points:
    layer = Points(
        data=np.random.normal(size=(2, 3)),
        features={
            napari_threedee.annotators.spheres.constants.SPHERE_ID_FEATURES_KEY: [0, 1],
            napari_threedee.annotators.spheres.constants.SPHERE_RADIUS_FEATURES_KEY: [5, 5]
        },
        metadata={N3D_METADATA_KEY: {
            ANNOTATION_TYPE_KEY: SphereAnnotator.ANNOTATION_TYPE
        }}
    )
    return layer


@pytest.fixture
def valid_n3d_sphere_layer() -> napari.layers.Points:
    return _valid_n3d_sphere_layer()


def _valid_n3d_sphere_zarr() -> zarr.Array:
    n3d_zarr = zarr.array(np.random.normal(size=(2, 3)))
    n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = SphereAnnotator.ANNOTATION_TYPE
    n3d_zarr.attrs[napari_threedee.annotators.spheres.constants.SPHERE_ID_FEATURES_KEY] = [0, 1]
    n3d_zarr.attrs[
        napari_threedee.annotators.spheres.constants.SPHERE_RADIUS_FEATURES_KEY] = [5, 5]
    return n3d_zarr


@pytest.fixture
def valid_n3d_sphere_zarr() -> zarr.Array:
    return _valid_n3d_sphere_zarr()


def _valid_n3d_point_layer() -> napari.layers.Points:
    layer = Points(
        data=np.random.normal(size=(2, 3)),
        metadata={N3D_METADATA_KEY: {
            ANNOTATION_TYPE_KEY: PointAnnotator.ANNOTATION_TYPE
        }}
    )
    return layer


@pytest.fixture
def valid_n3d_point_layer() -> napari.layers.Points:
    return _valid_n3d_point_layer()


def _valid_n3d_point_zarr() -> zarr.Array:
    n3d_zarr = zarr.array(np.random.normal(size=(2, 3)))
    n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = PointAnnotator.ANNOTATION_TYPE
    return n3d_zarr


@pytest.fixture
def valid_n3d_point_zarr() -> zarr.Array:
    return _valid_n3d_point_zarr()