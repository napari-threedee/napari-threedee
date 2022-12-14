# Point Specification

## Layer Specification
Point annotations are stored on a napari 
[`Points` layer](https://napari.org/stable/howtos/layers/points.html).

- `Points.data` is an `(n, d)` array of n d-dimensional points.
- `Points.metadata["n3d_metadata"]` is a dictionary with the following 
  key/value pairs
   - `annotation_type`: `point`



## Zarr Array Specification
The following assumes an `n3d` file has been read into a variable called 
`n3d` using the *zarr* library

```python
import zarr

zarr.load('points.n3d')
```

- `n3d` is an `(n, d)` [`zarr.core.Array`](https://zarr.readthedocs.
  io/en/stable/api/core.html) containing the points
- `n3d.attrs["annotation_type"]` is `"sphere"`


