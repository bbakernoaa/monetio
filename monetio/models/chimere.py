"""Chimere Reader. Deprecated wrapper — use monetio.load('chimere', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.chimere import ChimereReader  # noqa: F401


@deprecated_wrapper(
    "monetio.models.chimere.open_mfdataset",
    'monetio.load("chimere", files=...)',
)
def open_mfdataset(files, var_list=None, surf_only=False, **kwargs):
    """Method to open Chimere model netcdf output files.

    Parameters
    ----------
    files : list[str]
        files is a list of path(s) of the file(s).
    var_list: list[str]
        list of variable names meant to be kept for the analysis.
    surf_only: bool
        boolean flag specifying if only surface data (layer 0) should be kept for analysis.

    Returns
    -------
    xarray.Dataset
        Chimere model dataset in standard format for use
        in MELODIES-MONET
    """
    return ChimereReader().open_dataset(
        files=files, var_list=var_list, surf_only=surf_only, **kwargs
    )
