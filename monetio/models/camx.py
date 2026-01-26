"""
CAMx Reader. Redirection to monetio.readers.camx
"""

from ..readers.camx import (
    CAMxReader,
    add_lazy_nox,
    add_lazy_noy,
    add_lazy_pm10,
    add_lazy_pm25,
    add_lazy_pm_course,
    add_multiple_lazy,
    camx_preprocess,
    coarse,
    fine,
    noy_gas,
    poc,
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
