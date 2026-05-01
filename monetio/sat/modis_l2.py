"""MODIS L2 Reader. Deprecated wrapper — use monetio.load('modis_l2', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.modis_l2 import MODISL2Reader  # noqa: F401


@deprecated_wrapper(
    "monetio.sat.modis_l2.read_dataset",
    'monetio.load("modis_l2", files=...)',
)
def read_dataset(fname, variable_dict, **kwargs):
    """Read a single MODIS L2 HDF file.

    Parameters
    ----------
    fname : str
        Input file path.
    variable_dict : dict
        Variable configuration dictionary.
    **kwargs : dict
        Additional arguments forwarded to ``MODISL2Reader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return MODISL2Reader().open_dataset(files=fname, variable_dict=variable_dict, **kwargs)


@deprecated_wrapper(
    "monetio.sat.modis_l2.read_mfdataset",
    'monetio.load("modis_l2", files=...)',
)
def read_mfdataset(fnames, variable_dict, debug=False, **kwargs):
    """Read multiple MODIS L2 HDF files.

    Parameters
    ----------
    fnames : str
        Regular expression for input file paths.
    variable_dict : dict
        Variable configuration dictionary.
    debug : bool
        Enable debug logging.
    **kwargs : dict
        Additional arguments forwarded to ``MODISL2Reader.open_dataset``.

    Returns
    -------
    dict
        Ordered dict of granules keyed by datetime string.
    """
    return MODISL2Reader().open_dataset(
        files=fnames, variable_dict=variable_dict, debug=debug, **kwargs
    )


@deprecated_wrapper(
    "monetio.sat.modis_l2.apply_quality_flag",
    'monetio.load("modis_l2", files=...)',
)
def apply_quality_flag(ds, **kwargs):
    """Apply quality flag to a MODIS L2 dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with quality_flag attribute.
    **kwargs : dict
        Additional arguments.
    """
    return MODISL2Reader().apply_quality_flag(ds, **kwargs)
