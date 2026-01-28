"""
UFS-AQM File Reader. Redirection to monetio.readers.ufs
"""

from ..readers.ufs import (  # noqa: F401
    UFSReader,
    add_lazy_nox,
    add_lazy_noy_a,
    add_lazy_noy_g,
    add_lazy_pm10,
    add_lazy_pm25,
    dict_species_sums,
)


def open_mfdataset(fname, **kwargs):
    """Method to open RFFS-CMAQ dyn* netcdf files.

    Parameters
    ----------
    fname : string or list
        fname is the path to the file or files.  It will accept hot keys in
        strings as well.
    **kwargs : dict
        Additional arguments passed to UFSReader.open_dataset

    Returns
    -------
    xarray.DataSet
        UFS-AQM model dataset in standard format for use in MELODIES-MONET

    """
    return UFSReader().open_dataset(files=fname, **kwargs)
