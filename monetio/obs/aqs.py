"""AQS Reader. Deprecated wrapper — use monetio.load('aqs', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.aqs import AQS, AQSReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.aqs.add_data",
    'monetio.load("aqs", dates=...)',
)
def add_data(
    dates,
    param=None,
    daily=False,
    network=None,
    download=False,
    local=False,
    wide_fmt=True,
    n_procs=1,
    meta=False,
    as_xarray=True,
):
    """Retrieve and load AQS data."""
    return AQSReader().open_dataset(
        dates=dates,
        param=param,
        daily=daily,
        network=network,
        download=download,
        local=local,
        wide_fmt=wide_fmt,
        n_procs=n_procs,
        meta=meta,
        as_xarray=as_xarray,
    )
