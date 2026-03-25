"""
SKYNET Reader Redirection
"""

from ..readers.skynet import SKYNETReader


def add_data(
    dates=None,
    siteid=None,
    product="AOT",
    as_xarray=False,
    **kwargs,
):
    """Retrieve and load SKYNET data."""
    return SKYNETReader().open_dataset(
        dates=dates,
        siteid=siteid,
        product=product,
        as_xarray=as_xarray,
        **kwargs,
    )


def add_local(
    fname,
    as_xarray=False,
    **kwargs,
):
    """Read a local SKYNET file."""
    return SKYNETReader().open_dataset(
        files=fname,
        as_xarray=as_xarray,
        **kwargs,
    )
