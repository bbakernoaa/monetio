"""ISH Reader. Deprecated wrapper — use monetio.load('ish', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.ish import ISH, ISHReader, read_ish_file  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.ish.add_data",
    'monetio.load("ish", dates=...)',
)
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
    source="aws",
    as_xarray=True,
):
    """Retrieve and load ISH data."""
    return ISHReader().open_dataset(
        dates=dates,
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
        source=source,
        as_xarray=as_xarray,
    )
