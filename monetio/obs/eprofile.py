"""E-PROFILE Reader. Deprecated wrapper — use monetio.load('eprofile', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.eprofile import EPROFILEReader  # noqa: F401


@deprecated_wrapper(
    "monetio.obs.eprofile.add_data",
    'monetio.load("eprofile", files=...)',
)
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
