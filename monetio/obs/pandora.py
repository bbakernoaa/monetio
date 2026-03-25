"""
Pandora Reader Redirection
"""

from ..readers.pandora import PandoraReader


def add_data(
    dates=None,
    siteid=None,
    instrument=None,
    product="no2",
    as_xarray=True,
    **kwargs,
):
    """Retrieve and load Pandora data."""
    return PandoraReader().open_dataset(
        dates=dates,
        siteid=siteid,
        instrument=instrument,
        product=product,
        as_xarray=as_xarray,
        **kwargs,
    )


def add_local(
    fname,
    as_xarray=True,
    **kwargs,
):
    """Read a local Pandora file."""
    return PandoraReader().open_dataset(
        files=fname,
        as_xarray=as_xarray,
        **kwargs,
    )
