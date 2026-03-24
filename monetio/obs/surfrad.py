"""
SURFRAD Reader Redirection
"""

from ..readers.surfrad import SURFRADReader


def add_data(
    dates,
    *,
    sites=None,
    as_xarray=True,
    **kwargs,
):
    """Retrieve and load SURFRAD data."""
    return SURFRADReader().open_dataset(
        dates=dates,
        sites=sites,
        as_xarray=as_xarray,
        **kwargs,
    )
