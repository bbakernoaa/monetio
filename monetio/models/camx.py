"""CAMx Reader. Deprecated wrapper — use monetio.load('camx', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.camx import (  # noqa: F401
    CAMxReader,
    camx_preprocess,
    coarse,
    fine,
    noy_gas,
    poc,
)


@deprecated_wrapper(
    "monetio.models.camx.open_dataset",
    'monetio.load("camx", files=...)',
)
def open_dataset(fname, **kwargs):
    """Method to open CAMx files using pseudonetcdf.

    Parameters
    ----------
    fname : string or list
        fname is the path to the file or files.
    **kwargs : dict
        Additional arguments passed to CAMxReader.open_dataset

    Returns
    -------
    xarray.DataSet
    """
    return CAMxReader().open_dataset(files=fname, **kwargs)
