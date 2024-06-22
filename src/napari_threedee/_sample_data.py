from typing import List

import mrcfile
import numpy as np
from napari.types import LayerDataTuple
import pooch


def get_hiv_tomogram() -> np.ndarray:
    """Get the HIV tomogram as a numpy array.

    Data from: https://zenodo.org/records/6504891
    """
    tomogram_path = pooch.retrieve(
        url="doi:10.5281/zenodo.6504891/01_10.00Apx.mrc",
        known_hash="md5:426325d006fe04276ea01df9d83ad510",
        progressbar=True
    )
    return mrcfile.read(tomogram_path)


def hiv_sample_tomogram() -> List[LayerDataTuple]:
    """napari sample data function for hiv virus-like particles tomogram.

    Returns
    -------
    layer_data : List[LayerDataTuple]
        The data for the layers to be constructed.
        The LayerDataTuple has the following elements:
            - tomogram: the image
            - layer_kwargs: the keyword arguments passed to the
              napari add_image() method.
            - "image" the type of the layer
    """
    tomogram = get_hiv_tomogram()

    # keyword arguments for Viewer.add_image()
    layer_kwargs = {"name": "HIV tomogram", "colormap": "gray_r"}

    return [(tomogram, layer_kwargs, "image")]
