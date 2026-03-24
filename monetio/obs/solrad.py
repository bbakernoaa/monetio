"""
SOLRAD Reader Redirection
"""

from ..readers.solrad import SOLRADReader


def add_data(
    dates,
    *,
    sites=None,
    as_xarray=True,
    **kwargs,
):
    """Retrieve and load SOLRAD data."""
    return SOLRADReader().open_dataset(
        dates=dates,
        sites=sites,
        as_xarray=as_xarray,
        **kwargs,
    )
