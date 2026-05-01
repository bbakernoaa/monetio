"""ACTRIS/EBAS Reader. Deprecated wrapper — use monetio.load('actris', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.actris import ACTRISReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.actris.add_data",
    'monetio.load("actris", files=...)',
)
def add_data(
    files=None,
    **kwargs,
):
    """Retrieve and load ACTRIS/EBAS data."""
    return ACTRISReader().open_dataset(
        files=files,
        **kwargs,
    )
