"""
HYSPLIT Reader. Redirection to monetio.readers.hysplit
"""

from ..readers.hysplit import HYSPLITReader


def open_dataset(fname, **kwargs):
    """Method to open HYSPLIT netcdf files.

    Parameters
    ----------
    fname : string or list
        fname is the path to the file or files.
    **kwargs : dict
        Additional arguments passed to HYSPLITReader.open_dataset

    Returns
    -------
    xarray.DataSet
    """
    return HYSPLITReader().open_dataset(files=fname, **kwargs)
