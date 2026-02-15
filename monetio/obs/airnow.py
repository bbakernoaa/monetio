"""
AirNow Reader. Redirection to monetio.readers.airnow
"""

from ..readers.airnow import AirNowReader, build_urls, get_utcoffset, retrieve  # noqa: F401


def add_data(
    dates,
    *,
    download=False,
    wide_fmt=True,
    n_procs=1,
    daily=False,
    bad_utcoffset="drop",
    as_xarray=True,
):
    """Retrieve and load AirNow data."""
    return AirNowReader().open_dataset(
        dates=dates,
        download=download,
        wide_fmt=wide_fmt,
        n_procs=n_procs,
        daily=daily,
        bad_utcoffset=bad_utcoffset,
        as_xarray=as_xarray,
    )


def aggregate_files(dates, *, download=False, n_procs=1, daily=False, bad_utcoffset="drop"):
    """Aggregate AirNow files."""
    return AirNowReader().open_dataset(
        dates=dates,
        download=download,
        wide_fmt=False,
        n_procs=n_procs,
        daily=daily,
        bad_utcoffset=bad_utcoffset,
    )
