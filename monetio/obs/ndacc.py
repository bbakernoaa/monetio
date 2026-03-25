"""
NDACC Reader Redirection
"""

from ..readers.ndacc import NDACCReader


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
