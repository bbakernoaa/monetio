"""
E-PROFILE Reader Redirection
"""

from ..readers.eprofile import EPROFILEReader


def add_data(
    files=None,
    dates=None,
    **kwargs,
):
    """Retrieve and load E-PROFILE data."""
    return EPROFILEReader().open_dataset(
        files=files,
        dates=dates,
        **kwargs,
    )
