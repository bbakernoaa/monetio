"""TEMPO L2 Reader. Deprecated wrapper — use monetio.load('tempo', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.tempo import TEMPOReader  # noqa: F401


@deprecated_wrapper(
    "monetio.sat.tempo_l2.open_dataset",
    'monetio.load("tempo", files=...)',
)
def open_dataset(fnames, variable_dict, debug=False, **kwargs):
    """Open one or more TEMPO L2 NO2 files.

    Parameters
    ----------
    fnames : str
        Glob expression for input file paths.
    variable_dict : dict or str or sequence
        Variable configuration.
    debug : bool
        Enable debug logging.
    **kwargs : dict
        Additional arguments forwarded to ``TEMPOReader.open_dataset``.

    Returns
    -------
    dict
        Dict mapping reference time string to xarray.Dataset granules.
    """
    return TEMPOReader().open_dataset(
        files=fnames, variable_dict=variable_dict, debug=debug, **kwargs
    )
