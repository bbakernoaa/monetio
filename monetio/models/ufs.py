"""UFS-AQM File Reader. Deprecated wrapper — use monetio.load('ufs', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.ufs import UFSReader  # noqa: F401


@deprecated_wrapper(
    "monetio.models.ufs.open_mfdataset",
    'monetio.load("ufs", files=...)',
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
