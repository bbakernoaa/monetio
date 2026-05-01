"""NDACC Reader. Deprecated wrapper — use monetio.load('ndacc', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.ndacc import NDACCReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.ndacc.add_data",
    'monetio.load("ndacc", dates=...)',
)
def add_data(
    dates=None,
    siteid=None,
    instrument=None,
    as_xarray=True,
    **kwargs,
):
    """Retrieve and load NDACC data."""
    return NDACCReader().open_dataset(
        dates=dates,
        siteid=siteid,
        instrument=instrument,
        as_xarray=as_xarray,
        **kwargs,
    )


@deprecated_wrapper(
    "monetio.obs.ndacc.add_local",
    'monetio.load("ndacc", files=...)',
)
def add_local(
    fname,
    as_xarray=True,
    **kwargs,
):
    """Read a local NDACC file."""
    return NDACCReader().open_dataset(
        files=fname,
        as_xarray=as_xarray,
        **kwargs,
    )
