"""RAQMS Reader. Deprecated wrapper — use monetio.load('raqms', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.raqms import RAQMSReader  # noqa: F401


@deprecated_wrapper(
    "monetio.models.raqms.open_dataset",
    'monetio.load("raqms", files=...)',
)
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


@deprecated_wrapper(
    "monetio.models.raqms.open_mfdataset",
    'monetio.load("raqms", files=...)',
)
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
