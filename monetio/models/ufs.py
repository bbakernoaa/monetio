"""
UFS-AQM File Reader. Redirection to monetio.readers.ufs
"""

from ..readers.ufs import UFSReader


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
