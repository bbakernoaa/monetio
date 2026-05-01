"""Pandora Reader. Deprecated wrapper — use monetio.load('pandora', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.pandora import PandoraReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.pandora.add_data",
    'monetio.load("pandora", dates=...)',
)
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


@deprecated_wrapper(
    "monetio.obs.pandora.add_local",
    'monetio.load("pandora", files=...)',
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
