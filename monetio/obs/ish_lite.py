"""
ISH Lite Reader Redirection
"""

from ..readers.ish_lite import ISH, ISHLiteReader


def add_data(
    dates,
    *,
    box=None,
    country=None,
    state=None,
    site=None,
    resample=False,
    window="h",
    n_procs=1,
    verbose=False,
):
    """Retrieve and load ISH-lite data as a DataFrame."""
    return ISHLiteReader().open_dataset(
        dates,
        box=box,
        country=country,
        state=state,
        site=site,
        resample=resample,
        window=window,
        n_procs=n_procs,
        verbose=verbose,
    )
