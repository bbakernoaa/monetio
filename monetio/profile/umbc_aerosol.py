"""UMBC Aerosol (CL51) Reader. Deprecated wrapper — use monetio.load('umbc_aerosol', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.umbc_aerosol import UMBCAerosolReader  # noqa: F401


@deprecated_wrapper(
    "monetio.profile.umbc_aerosol.open_dataset",
    'monetio.load("umbc_aerosol", files=...)',
)
def open_dataset(fname, **kwargs):
    """Open a single UMBC Aerosol (CL51 Ceilometer) HDF5 file.

    Parameters
    ----------
    fname : str
        Path to the HDF5 file.
    **kwargs : dict
        Additional arguments forwarded to ``UMBCAerosolReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return UMBCAerosolReader().open_dataset(files=fname, **kwargs)


@deprecated_wrapper(
    "monetio.profile.umbc_aerosol.open_mfdataset",
    'monetio.load("umbc_aerosol", files=...)',
)
def open_mfdataset(fname, **kwargs):
    """Open multiple UMBC Aerosol (CL51 Ceilometer) HDF5 files.

    Parameters
    ----------
    fname : str
        Glob pattern or path to HDF5 files.
    **kwargs : dict
        Additional arguments forwarded to ``UMBCAerosolReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return UMBCAerosolReader().open_dataset(files=fname, **kwargs)
