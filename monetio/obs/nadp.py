"""NADP Reader. Deprecated wrapper — use monetio.load('nadp', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.nadp import NADPReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.nadp.add_data",
    'monetio.load("nadp", dates=...)',
)
def add_data(dates, network="NTN", siteid=None, weekly=True, as_xarray=True):
    """Retrieve and load NADP data."""
    return NADPReader().open_dataset(
        dates,
        network=network,
        siteid=siteid,
        weekly=weekly,
        as_xarray=as_xarray,
    )
