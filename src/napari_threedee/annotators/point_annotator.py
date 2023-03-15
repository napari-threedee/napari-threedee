import os
from typing import Optional, Union

import napari.layers
import napari.types
import numpy as np
import zarr
from pydantic import validator

from napari_threedee._backend.threedee_model import ThreeDeeModel
from .base import N3dDataModel
from .constants import N3D_METADATA_KEY, ANNOTATION_TYPE_KEY, POINT_ANNOTATION_TYPE_KEY
from ..mouse_callbacks import add_point_on_plane
from napari_threedee.utils.napari_utils import add_mouse_callback_safe, \
    remove_mouse_callback_safe


def validate_layer(layer: napari.layers.Points):
    """Ensure a napari points layer matches the n3d layer specification."""
    if N3D_METADATA_KEY not in layer.metadata:
        raise ValueError(f"{N3D_METADATA_KEY} not found")
    n3d_metadata = layer.metadata[N3D_METADATA_KEY]
    if n3d_metadata[ANNOTATION_TYPE_KEY] != POINT_ANNOTATION_TYPE_KEY:
        raise ValueError("Cannot read as n3d points layer.")


def validate_n3d_zarr(n3d_zarr: zarr.Array):
    """Ensure an n3d zarr array contains data for n3d points layer."""
    if ANNOTATION_TYPE_KEY not in n3d_zarr.attrs:
        raise ValueError("cannot read as n3d points.")
    if n3d_zarr.attrs[ANNOTATION_TYPE_KEY] != POINT_ANNOTATION_TYPE_KEY:
        raise ValueError("cannot read as n3d points.")


class N3dPoints(N3dDataModel):
    data: np.ndarray

    @classmethod
    def from_layer(cls, layer: napari.layers.Points):
        return cls(data=layer.data)

    def as_layer(self) -> napari.layers.Points:
        metadata = {N3D_METADATA_KEY: {ANNOTATION_TYPE_KEY: POINT_ANNOTATION_TYPE_KEY}}
        layer = napari.layers.Points(data=self.data, metadata=metadata)
        validate_layer(layer)
        return layer

    @classmethod
    def from_n3d_zarr(cls, path: os.PathLike):
        n3d_zarr = zarr.open(path)
        validate_n3d_zarr(n3d_zarr)
        return cls(data=np.array(n3d_zarr))

    def to_n3d_zarr(self, path: os.PathLike):
        n3d_zarr = zarr.open_array(
            store=path,
            shape=self.data.shape,
            dtype=self.data.dtype,
            mode="w"
        )
        n3d_zarr[...] = self.data
        n3d_zarr.attrs[ANNOTATION_TYPE_KEY] = POINT_ANNOTATION_TYPE_KEY
        return n3d_zarr

    @validator('data', pre=True)
    def ensure_2d_float32_array(cls, value):
        return np.atleast_2d(np.asarray(value, dtype=np.float32))


class PointAnnotator(ThreeDeeModel):
    def __init__(
        self,
        viewer: napari.Viewer,
        image_layer: Optional[napari.layers.Image] = None,
        points_layer: Optional[napari.layers.Points] = None,
        enabled: bool = False
    ):
        self.viewer = viewer
        self.points_layer = points_layer
        self.image_layer = image_layer
        self.enabled = enabled

    def _mouse_callback(self, viewer, event):
        if (self.image_layer is None) or (self.points_layer is None):
            return
        add_point_on_plane(
            viewer=viewer,
            event=event,
            points_layer=self.points_layer,
            plane_layer=self.image_layer
        )

    def set_layers(
        self,
        image_layer: napari.layers.Image,
        points_layer: napari.layers.Points
    ):
        validate_layer(points_layer)
        self.image_layer = image_layer
        self.points_layer = points_layer

    def _on_enable(self):
        add_mouse_callback_safe(
            self.viewer.mouse_drag_callbacks, self._mouse_callback
        )

    def _on_disable(self):
        remove_mouse_callback_safe(
            self.viewer.mouse_drag_callbacks, self._mouse_callback
        )
