"""
ACTRIS/EBAS Reader Redirection
"""

from ..readers.actris import ACTRISReader


def add_data(
    files=None,
    **kwargs,
):
    """Retrieve and load ACTRIS/EBAS data."""
    return ACTRISReader().open_dataset(
        files=files,
        **kwargs,
    )
