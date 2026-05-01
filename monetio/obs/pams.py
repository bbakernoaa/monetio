"""PAMS Reader. Deprecated wrapper — use monetio.load('pams', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.pams import PAMSReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.pams.add_data",
    'monetio.load("pams", files=...)',
)
def add_data(filename, as_xarray=True):
    """Retrieve and load PAMS data."""
    return PAMSReader().open_dataset(files=filename, as_xarray=as_xarray)
