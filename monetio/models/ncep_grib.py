"""NCEP GRIB Reader. Deprecated wrapper — use monetio.load('ncep_grib', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.ncep_grib import NCEPGribReader  # noqa: F401


@deprecated_wrapper(
    "monetio.models.ncep_grib.open_dataset",
    'monetio.load("ncep_grib", files=...)',
)
def open_dataset(fname, **kwargs):
    """Open a single dataset from model outputs (grib2 currently)

    Parameters
    ----------
    fname : string
        Filename to be opened

    Returns
    -------
    xarray.Dataset
    """
    return NCEPGribReader().open_dataset(files=fname, **kwargs)


@deprecated_wrapper(
    "monetio.models.ncep_grib.open_mfdataset",
    'monetio.load("ncep_grib", files=...)',
)
def open_mfdataset(fname, **kwargs):
    """Open multiple files from model outputs (grib2 currently)

    Parameters
    ----------
    fname : string
        Filenames to be opened

    Returns
    -------
    xarray.Dataset
    """
    return NCEPGribReader().open_dataset(files=fname, **kwargs)
