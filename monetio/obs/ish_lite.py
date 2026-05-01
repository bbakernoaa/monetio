"""ISH Lite Reader. Deprecated wrapper — use monetio.load('ish_lite', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.ish_lite import ISH, ISHLiteReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.ish_lite.add_data",
    'monetio.load("ish_lite", dates=...)',
)
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
    as_xarray=True,
):
    """Retrieve and load ISH Lite data."""
    return ISHLiteReader().open_dataset(
        dates=dates,
        box=box,
        country=country,
        state=state,
        site=site,
        resample=resample,
        window=window,
        n_procs=n_procs,
        verbose=verbose,
        as_xarray=as_xarray,
    )
