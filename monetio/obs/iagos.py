"""IAGOS Reader. Deprecated wrapper — use monetio.load('iagos', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.iagos import IAGOSReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.iagos.add_data",
    'monetio.load("iagos", dates=...)',
)
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


@deprecated_wrapper(
    "monetio.obs.iagos.add_local",
    'monetio.load("iagos", files=...)',
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
