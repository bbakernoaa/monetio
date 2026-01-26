"""
FV3-Chem Reader. Redirection to monetio.readers.fv3chem
"""

from ..readers.fv3chem import FV3ChemReader


def open_dataset(fname, **kwargs):
    """Method to open FV3-Chem netcdf files.

    Parameters
    ----------
    fname : string or list
        fname is the path to the file or files.
    **kwargs : dict
        Additional arguments passed to FV3ChemReader.open_dataset

    Returns
    -------
    xarray.DataSet
    """
    return FV3ChemReader().open_dataset(files=fname, **kwargs)


def open_mfdataset(fname, **kwargs):
    """Method to open FV3-Chem netcdf files.

    Parameters
    ----------
    fname : string or list
        fname is the path to the file or files.
    **kwargs : dict
        Additional arguments passed to FV3ChemReader.open_dataset

    Returns
    -------
    xarray.DataSet
    """
    return FV3ChemReader().open_dataset(files=fname, **kwargs)
