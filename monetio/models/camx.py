"""
CAMx Reader. Redirection to monetio.readers.camx
"""

from ..readers.camx import CAMxReader


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
