# Input/Output

Annotation layers can be serialized to disk as `n3d` files. `n3d` files are
[`zarr` files](https://zarr.readthedocs.io/en/stable/index.html).

## *napari-threedee*
*napari-threedee* implements readers and writers for `n3d` files.

## Python
`n3d` files can be opened in Python with the 
[*zarr*](https://zarr.readthedocs.io/en/stable/index.html) 
library.

```python
import zarr

n3d_data = zarr.load('annotation.n3d')
```

These objects contain some attributes allowing them to be 
correctly interpreted by *napari-threedee*. The 
attribute `annotation_type` maps to a specific reader function 
*napari-threedee* will use to load the data.

```python
n3d_data.attrs["annotation_type"]
```
```ipython
Out[2]: 'spline'
```

## Other programming languages

For working with `n3d` (`zarr`) files in other languages, please look at the 
[zarr implementations repository](https://github.com/zarr-developers/zarr_implementations).
