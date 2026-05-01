"""TROPOMI L2 NO2 Reader. Deprecated wrapper — use monetio.load('tropomi', ...) instead."""

import warnings

from ..readers._deprecation import deprecated_wrapper
from ..readers.tropomi import TROPOMIReader  # noqa: F401


@deprecated_wrapper(
    "monetio.sat.tropomi_l2_no2.open_dataset",
    'monetio.load("tropomi", files=...)',
)
def open_dataset(fnames, variable_dict, debug=False, **kwargs):
    """Open one or more TROPOMI L2 NO2 files.

    Parameters
    ----------
    fnames : str
        Glob expression for input file paths.
    variable_dict : dict or str or sequence
        Variable configuration.
    debug : bool
        Enable debug logging.
    **kwargs : dict
        Additional arguments forwarded to ``TROPOMIReader.open_dataset``.

    Returns
    -------
    OrderedDict
        Dict mapping reference time string to list of xarray.Dataset granules.
    """
    return TROPOMIReader().open_dataset(
        files=fnames, variable_dict=variable_dict, debug=debug, **kwargs
    )


def read_trpdataset(*args, **kwargs):
    """Alias for :func:`open_dataset`."""
    warnings.warn(
        "read_trpdataset is an alias for open_dataset and may be removed in the future",
        FutureWarning,
        stacklevel=2,
    )
    return open_dataset(*args, **kwargs)
