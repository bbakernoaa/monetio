"""CRN Reader. Deprecated wrapper — use monetio.load('crn', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.crn import CRNReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.crn.add_data",
    'monetio.load("crn", dates=...)',
)
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
