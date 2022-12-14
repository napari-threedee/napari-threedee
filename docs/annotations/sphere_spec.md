# Sphere Specification

## Layer Specification
Sphere annotations are stored on a napari 
[`Points` layer](https://napari.org/stable/howtos/layers/points.html) and 
can contain multiple splines.

- `Points.data` is an `(n, d)` array of n d-dimensional points.
- `Points.features` is a table which will contain column called 
    - `sphere_id`, an integer id for each sphere.
    - `radius`, a radius for each sphere.
- `Points.metadata["n3d_metadata"]` is a dictionary with the following 
  key/value pairs
   - `annotation_type`: `sphere`



## Zarr Array Specification
The following assumes an `n3d` file has been read into a variable called 
`n3d` using the *zarr* library

```python
import zarr

zarr.load('sphere.n3d')
```

- `n3d` is an `(n, d)` [`zarr.core.Array`](https://zarr.readthedocs.
  io/en/stable/api/core.html) containing the center of each sphere
- `n3d.attrs["annotation_type"]` is `"sphere"`
- `n3d.attrs["sphere_id"]` is an `(n, )` array containing a unique id for 
  each sphere
- `n3d.attrs["radius"]` is an `(n, )` array containing the radius for each 
  sphere


