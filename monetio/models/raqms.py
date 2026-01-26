"""
Reader for RAQMS real-time files. Redirection to monetio.readers.raqms
"""

from ..readers.raqms import RAQMSReader


def open_dataset(fname, **kwargs):
    """Open a single dataset from RAQMS output. Currently expects netCDF file format.

    Parameters
    ----------
    fname : str
        File to be opened.
    **kwargs : dict
        Additional arguments passed to RAQMSReader.open_dataset

    Returns
    -------
    xarray.Dataset
    """
    return RAQMSReader().open_dataset(files=fname, **kwargs)


def open_mfdataset(fname, **kwargs):
    """Open a multiple file dataset from RAQMS output.

    Parameters
    ----------
    fname : str or list of str
        Files to be opened, expressed as a glob string or list of string paths.
    **kwargs : dict
        Additional arguments passed to RAQMSReader.open_dataset

    Returns
    -------
    xarray.Dataset
    """
    return RAQMSReader().open_dataset(files=fname, **kwargs)


def _ensure_mfdataset_filenames(fname):
    """Checks if RAQMS netcdf dataset

    Parameters
    ----------
    fname : str or list of str

    Returns
    -------
    list of str
        The file paths.
    bool
        Whether all of files are the expected uwhyb netCDF format.
    """
    from glob import glob
    from os.path import basename

    if isinstance(fname, str):
        fpaths = sorted(glob(fname))
    else:
        fpaths = sorted(fname)

    # Check file name is of the expected format
    good = len(fpaths) > 0 and all(
        fp.endswith(".nc") and "uwhyb" in basename(fp) for fp in fpaths
    )

    return fpaths, good
