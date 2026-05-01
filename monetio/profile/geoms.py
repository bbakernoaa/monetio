"""GEOMS Profile Reader. Deprecated wrapper — use monetio.load('geoms', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.geoms import GEOMSReader  # noqa: F401


@deprecated_wrapper(
    "monetio.profile.geoms.open_dataset",
    'monetio.load("geoms", files=...)',
)
def open_dataset(fp, *, rename_all=True, squeeze=True, **kwargs):
    """Open a file in GEOMS format, e.g. modern TOLNet files.

    Parameters
    ----------
    fp : str
        File path.
    rename_all : bool, default: True
        Rename all non-coordinate variables.
    squeeze : bool, default: True
        Apply ``.squeeze()`` before returning the Dataset.
    **kwargs : dict
        Additional arguments forwarded to ``GEOMSReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return GEOMSReader().open_dataset(
        files=fp, rename_all=rename_all, squeeze=squeeze, **kwargs
    )
