"""
MPLNET Reader Redirection
"""

from ..readers.mplnet import MPLNETReader


def add_data(
    files,
    **kwargs,
):
    """Retrieve and load MPLNET data."""
    return MPLNETReader().open_dataset(
        files=files,
        **kwargs,
    )
