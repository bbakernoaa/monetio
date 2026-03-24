"""
EARLINET Reader Redirection
"""

from ..readers.earlinet import EARLINETReader


def add_data(
    files,
    **kwargs,
):
    """Retrieve and load EARLINET data."""
    return EARLINETReader().open_dataset(
        files=files,
        **kwargs,
    )
