"""
PAMS Reader Redirection
"""

from ..readers.pams import PAMSReader  # noqa: F401


def add_data(filename, as_xarray=True):
    """Retrieve and load PAMS data."""
    return PAMSReader().open_dataset(files=filename, as_xarray=as_xarray)
