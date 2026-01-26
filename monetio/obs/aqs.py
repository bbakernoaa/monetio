"""
AQS Reader Redirection
"""

from ..readers.aqs import AQS, AQSReader


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
):
    """Retrieve and load AQS data."""
    return AQSReader().open_dataset(
        dates,
        param=param,
        daily=daily,
        network=network,
        download=download,
        local=local,
        wide_fmt=wide_fmt,
        n_procs=n_procs,
        meta=meta,
    )
