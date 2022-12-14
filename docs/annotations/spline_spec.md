# Spline Specification

## Layer Specification
Spline annotations are stored on a napari 
[`Points` layer](https://napari.org/stable/howtos/layers/points.html) and 
can contain multiple splines.

- `Points.data` is an `(n, d)` array of n d-dimensional points.
- `Points.features` is a table which will contain a column called 
  `spline_id`, an integer id for each point.
- `Points.metadata["n3d_metadata"]` is a dictionary with the following 
  key/value pairs
   - `annotation_type`: `spline`

## Zarr Array Specification
The following assumes an `n3d` file has been read into a variable called 
`n3d` using the *zarr* library

```python
import zarr

zarr.load('spline.n3d')
```

- `n3d` is an `(n, d)` [`zarr.core.Array`](https://zarr.readthedocs.
  io/en/stable/api/core.html)
- `n3d.attrs["annotation_type"]` is `"spline"`
- `n3d.attrs["spline_id"]` is an `(n, )` array containing the spline id of 
  each point


