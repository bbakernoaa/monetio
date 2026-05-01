"""CMAQ File Reader. Deprecated wrapper — use monetio.load('cmaq', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.cmaq import CMAQReader  # noqa: F401


@deprecated_wrapper(
    "monetio.models.cmaq.open_dataset",
    'monetio.load("cmaq", files=...)',
)
def open_dataset(fname, **kwargs):
    """Method to open CMAQ IOAPI netcdf files.

    Parameters
    ----------
    fname : string or list
        fname is the path to the file or files.
    **kwargs : dict
        Additional arguments passed to CMAQReader.open_dataset

    Returns
    -------
    xarray.DataSet
    """
    return CMAQReader().open_dataset(files=fname, **kwargs)


@deprecated_wrapper(
    "monetio.models.cmaq.open_mfdataset",
    'monetio.load("cmaq", files=...)',
)
def open_mfdataset(fname, **kwargs):
    """Method to open CMAQ IOAPI netcdf files.

    Parameters
    ----------
    fname : string or list
        fname is the path to the file or files.
    **kwargs : dict
        Additional arguments passed to CMAQReader.open_dataset

    Returns
    -------
    xarray.DataSet
    """
    return CMAQReader().open_dataset(files=fname, **kwargs)
