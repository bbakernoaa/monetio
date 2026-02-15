"""
IMPROVE Reader Redirection
"""

from ..readers.improve import IMPROVEReader  # noqa: F401


def add_data(fname, add_meta=False, delimiter="\t", as_xarray=True):
    """Retrieve and load IMPROVE data."""
    return IMPROVEReader().open_dataset(
        files=fname, add_meta=add_meta, delimiter=delimiter, as_xarray=as_xarray
    )
