"""
CRN Reader. Redirection to monetio.readers.crn
"""

from ..readers.crn import CRN, CRNReader


def add_data(
    dates, param=None, daily=False, sub_hourly=False, download=False, latlonbox=None
):
    """Retrieve and load CRN data as a DataFrame."""
    return CRNReader().open_dataset(
        dates=dates,
        daily=daily,
        sub_hourly=sub_hourly,
        download=download,
        latlonbox=latlonbox,
    )
