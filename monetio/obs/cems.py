"""CEMS Reader. Deprecated wrapper — use monetio.load('cems', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.cems import CEMS, CEMSReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.cems.add_data",
    'monetio.load("cems", rdate=...)',
)
def add_data(rdate=None, states=["md"], download=False, verbose=True, files=None, as_xarray=True):
    """Retrieve and load CEMS data."""
    return CEMSReader().open_dataset(
        rdate=rdate,
        states=states,
        download=download,
        verbose=verbose,
        files=files,
        as_xarray=as_xarray,
    )
