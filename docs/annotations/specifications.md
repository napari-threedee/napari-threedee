# Overview

Annotations made using the annotators from *napari-threedee* are stored 
on a single layer. This layer will match a **specification** for a given
annotation type.

Per-point attributes will be stored in the layer features table. Other 
metadata will be stored in the layer metadata as a dictionary under the 
`"n3d_metadata"` key.

## Specifications

- [spline specification](./spline_spec.md) (points layer)
- [sphere specification](./sphere_spec.md) (points layer)
- [point specification](./point_spec.md) (points layer)
