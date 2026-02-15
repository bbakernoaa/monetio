"""
CRN Reader. Redirection to monetio.readers.crn
"""

from ..readers.crn import CRN, CRNReader  # noqa: F401


def add_data(
    dates,
    param=None,
    daily=False,
    sub_hourly=False,
    download=False,
    latlonbox=None,
    as_xarray=True,
):
    """Retrieve and load CRN data."""
    return CRNReader().open_dataset(
        dates=dates,
        daily=daily,
        sub_hourly=sub_hourly,
        download=download,
        latlonbox=latlonbox,
        as_xarray=as_xarray,
    )
