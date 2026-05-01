"""MOPITT L3 Reader. Deprecated wrapper — use monetio.load('mopitt', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.mopitt import MOPITTReader  # noqa: F401


@deprecated_wrapper(
    "monetio.sat.mopitt_l3.open_dataset",
    'monetio.load("mopitt", files=...)',
)
def open_dataset(files, varnames, **kwargs):
    """Open MOPITT level 3 data files.

    Parameters
    ----------
    files : str or Path or list
        Input file path(s).
    varnames : str or list of str
        The variable(s) to load from the MOPITT file.
    **kwargs : dict
        Additional arguments forwarded to ``MOPITTReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return MOPITTReader().open_dataset(
        files=files, varnames=varnames, **kwargs
    )
