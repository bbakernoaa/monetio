"""
ISH Reader Redirection
"""

from ..readers.ish import ISH, ISHReader


def add_data(
    dates,
    *,
    box=None,
    country=None,
    state=None,
    site=None,
    resample=True,
    window="h",
    download=False,
    n_procs=1,
    request_timeout=10,
    request_retries=4,
    verbose=False,
):
    """Retrieve and load ISH data as a DataFrame."""
    return ISHReader().open_dataset(
        dates,
        box=box,
        country=country,
        state=state,
        site=site,
        resample=resample,
        window=window,
        download=download,
        n_procs=n_procs,
        request_timeout=request_timeout,
        request_retries=request_retries,
        verbose=verbose,
    )
