"""IMPROVE Reader. Deprecated wrapper — use monetio.load('improve', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.improve import IMPROVEReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.improve.add_data",
    'monetio.load("improve", files=...)',
)
def add_data(fname, add_meta=False, delimiter="\t", as_xarray=True):
    """Retrieve and load IMPROVE data."""
    return IMPROVEReader().open_dataset(
        files=fname, add_meta=add_meta, delimiter=delimiter, as_xarray=as_xarray
    )
