import os
from abc import abstractmethod

import napari
from pydantic import BaseModel


class N3dDataModel(BaseModel):
    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    @classmethod
    @abstractmethod
    def from_layer(cls, layer: napari.layers.Layer):
        pass

    @abstractmethod
    def as_layer(self) -> napari.layers.Layer:
        pass

    @classmethod
    @abstractmethod
    def from_n3d_zarr(self, path: os.PathLike):
        pass

    @abstractmethod
    def to_n3d_zarr(self, path: os.PathLike) -> None:
        pass
