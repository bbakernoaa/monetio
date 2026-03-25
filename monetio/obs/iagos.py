"""
IAGOS Reader Redirection
"""

from ..readers.iagos import IAGOSReader


def add_data(
    dates=None,
    as_xarray=True,
    **kwargs,
):
    """Retrieve and load IAGOS data."""
    return IAGOSReader().open_dataset(
        dates=dates,
        as_xarray=as_xarray,
        **kwargs,
    )


def add_local(
    fname,
    as_xarray=True,
    **kwargs,
):
    """Read a local IAGOS file."""
    return IAGOSReader().open_dataset(
        files=fname,
        as_xarray=as_xarray,
        **kwargs,
    )
