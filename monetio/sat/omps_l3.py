"""OMPS L3 Reader. Deprecated wrapper — use monetio.load('omps', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.omps import OMPSReader  # noqa: F401


@deprecated_wrapper(
    "monetio.sat.omps_l3.open_dataset",
    'monetio.load("omps", files=...)',
)
def open_dataset(files, **kwargs):
    """Open OMPS nadir mapper Total Column Ozone L3 files.

    Parameters
    ----------
    files : str or Path or list
        Input file path(s).
    **kwargs : dict
        Additional arguments forwarded to ``OMPSReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return OMPSReader().open_dataset(files=files, **kwargs)
