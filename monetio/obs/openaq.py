"""
OpenAQ Reader Redirection
"""

from ..readers.openaq import OPENAQ, OpenAQReader, read_json, read_json2  # noqa: F401


def add_data(dates, *, n_procs=1, wide_fmt=True, as_xarray=True):
    """Add OpenAQ data from the OpenAQ S3 bucket."""
    return OpenAQReader().open_dataset(
        dates=dates,
        n_procs=n_procs,
        wide_fmt=wide_fmt,
        as_xarray=as_xarray,
    )
