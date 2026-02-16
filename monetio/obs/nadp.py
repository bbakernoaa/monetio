"""
NADP Reader Redirection
"""

from ..readers.nadp import NADPReader  # noqa: F401


def add_data(dates, network="NTN", siteid=None, weekly=True, as_xarray=True):
    """Retrieve and load NADP data."""
    return NADPReader().open_dataset(
        dates,
        network=network,
        siteid=siteid,
        weekly=weekly,
        as_xarray=as_xarray,
    )
