"""OpenAQ Reader. Deprecated wrapper — use monetio.load('openaq', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.openaq import OPENAQ, OpenAQReader, read_json, read_json2  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.openaq.add_data",
    'monetio.load("openaq", dates=...)',
)
def add_data(dates, *, n_procs=1, wide_fmt=True, as_xarray=True):
    """Add OpenAQ data from the OpenAQ S3 bucket."""
    return OpenAQReader().open_dataset(
        dates=dates,
        wide_fmt=wide_fmt,
        as_xarray=as_xarray,
    )
