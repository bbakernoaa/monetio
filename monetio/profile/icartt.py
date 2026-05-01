"""ICARTT Profile Reader. Deprecated wrapper — use monetio.load('icartt', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.icartt import ICARTTReader  # noqa: F401


@deprecated_wrapper(
    "monetio.profile.icartt.add_data",
    'monetio.load("icartt", files=...)',
)
def add_data(filename, **kwargs):
    """Open an ICARTT file and return an xarray Dataset.

    Parameters
    ----------
    filename : str
        Path to the ICARTT file.
    **kwargs : dict
        Additional arguments forwarded to ``ICARTTReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return ICARTTReader().open_dataset(files=filename, **kwargs)
