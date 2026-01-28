"""
CMAQ File Reader. Redirection to monetio.readers.cmaq
"""

from ..readers.cmaq import CMAQReader  # noqa: F401


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
